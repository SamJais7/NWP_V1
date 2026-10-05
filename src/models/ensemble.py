"""
Out-of-Fold (OOF) Meta-Ensemble Stacking & Dual Output Calibration.
Combines Rung 1 (Climatology), Rung 2 (Ensemble Spread), Rung 3 (Analogs),
Rung 4 (GBDT), and Rung 5 (Spatial U-Net) with Isotonic Regression Calibration.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.tabular_model import TabularBustClassifier
from src.data.feature_engineer import ClimatologicalBaseline, EnsembleSpreadBaseline, HistoricalAnalogEngine


class MetaEnsembleStacker:
    """
    Rung 6: Meta-Ensemble Stacking and Probability Calibration Engine.
    Outputs:
      1. Calibrated Confidence Index (0–100%) - Monotonically degrades with lead time
      2. Calibrated Bust Probability (0.0–1.0) - Flow-dependent risk
    """

    def __init__(self):
        self.climatology = ClimatologicalBaseline()
        self.spread_engine = EnsembleSpreadBaseline()
        self.analog_engine = HistoricalAnalogEngine()
        self.tabular_model = TabularBustClassifier()
        
        # Meta-learner on stacked OOF probabilities
        self.meta_learner = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced")
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        self.is_meta_fitted = False
        self.is_calibrated = False
        self.learned_weights: Optional[np.ndarray] = None

    def fit_meta(self, X_meta: np.ndarray, y_true: np.ndarray):
        """Fits the LogisticRegression meta-learner on genuine out-of-fold base predictions."""
        self.meta_learner.fit(X_meta, y_true)
        self.is_meta_fitted = True
        if hasattr(self.meta_learner, "coef_"):
            self.learned_weights = self.meta_learner.coef_[0]

    def calibrate(self, oof_preds: np.ndarray, y_true: np.ndarray):
        """Fits isotonic calibrator strictly on out-of-fold validation predictions."""
        self.calibrator.fit(oof_preds, y_true)
        self.is_calibrated = True

    def predict(
        self,
        df_features: pd.DataFrame,
        spatial_probs: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Executes stacked inference and produces dual calibrated outputs.
        Returns DataFrame with confidence_index, bust_probability, and alert_level.
        """
        n_samples = len(df_features)
        
        # 1. Rung 1: Climatological
        p_clim = np.array([
            self.climatology.predict_bust_probability(s, d)
            for s, d in zip(df_features["sub_id"].values, df_features["lead_day"].values)
        ])

        # 2. Rung 2: Ensemble Spread
        spread_vals = df_features["apcp_spread_mean"].values if "apcp_spread_mean" in df_features else np.ones(n_samples) * 4.0
        p_spread = np.array([
            self.spread_engine.predict_bust_probability(s, d)
            for s, d in zip(spread_vals, df_features["lead_day"].values)
        ])

        # 3. Rung 3: Analogs (Vectorized)
        analog_cols = [
            "somali_llj_index", "monsoon_trough_lat", "wd_depth", "apcp_mean",
            "apcp_spread_mean", "mjo_amplitude" if "mjo_amplitude" in df_features else "mjo_rmm1",
            "bsiso_index_1" if "bsiso_index_1" in df_features else "bsiso_index",
            "shear_mean"
        ]
        analog_mat = np.zeros((n_samples, 8))
        for col_idx, c in enumerate(analog_cols):
            if c in df_features:
                analog_mat[:, col_idx] = df_features[c].values
            else:
                analog_mat[:, col_idx] = 10.0 if col_idx == 0 else 0.0
        p_analog = self.analog_engine.match_analogs_batch(analog_mat)

        # 4. Rung 4: Tabular GBDT
        p_gbdt = self.tabular_model.predict_proba(df_features)

        # 5. Rung 5: Learned Meta-Stacker
        p_spatial = spatial_probs if spatial_probs is not None and len(spatial_probs) == n_samples else p_gbdt
        if self.is_meta_fitted:
            if spatial_probs is not None and len(spatial_probs) == n_samples and getattr(self.meta_learner, "n_features_in_", 4) == 5:
                X_meta = np.column_stack([p_clim, p_spread, p_analog, p_gbdt, p_spatial])
            else:
                X_meta = np.column_stack([p_clim, p_spread, p_analog, p_gbdt])
            stacked_raw = self.meta_learner.predict_proba(X_meta)[:, 1]
        else:
            # Baseline prior blend across Rungs 1 to 4
            stacked_raw = (
                0.10 * p_clim +
                0.20 * p_spread +
                0.25 * p_analog +
                0.45 * p_gbdt
            )
            if spatial_probs is not None and len(spatial_probs) == n_samples:
                stacked_raw = 0.85 * stacked_raw + 0.15 * spatial_probs

        # Isotonic Calibration
        if self.is_calibrated:
            cal_pred = self.calibrator.predict(stacked_raw)
            calibrated_bust_prob = np.clip(0.3 * stacked_raw + 0.7 * cal_pred, 0.02, 0.98)
        else:
            calibrated_bust_prob = np.clip(stacked_raw, 0.03, 0.97)

        # Dual Output 1: Calibrated Confidence Index (0–100%)
        # Monotonically degrades from Day 1 to Day 10
        # Confidence = 100 * (1 - Bust_Prob) * Degradation_Curve
        leads = df_features["lead_day"].values
        b_probs = calibrated_bust_prob
        max_possible_confs = 98.0 - 4.2 * (leads - 1)
        raw_confs = max_possible_confs * (1.0 - 0.75 * b_probs)
        conf_scores = np.round(np.clip(raw_confs, 5.0, 98.0), 1)
        b_probs_rounded = np.round(b_probs, 3)

        alert_levels = np.where(
            (b_probs_rounded >= 0.75) | (conf_scores < 25.0), "CRITICAL",
            np.where(
                (b_probs_rounded >= 0.50) | (conf_scores < 50.0), "LOW",
                np.where(
                    (b_probs_rounded >= 0.25) | (conf_scores < 75.0), "MODERATE", "HIGH"
                )
            )
        )
        alert_colors = np.where(
            alert_levels == "CRITICAL", "#f43f5e",
            np.where(alert_levels == "LOW", "#f97316",
                     np.where(alert_levels == "MODERATE", "#eab308", "#10b981"))
        )
        badges = np.where(
            alert_levels == "CRITICAL", "CRITICAL BUST WARNING",
            np.where(alert_levels == "LOW", "LOW CONFIDENCE ALERT",
                     np.where(alert_levels == "MODERATE", "MODERATE WATCH", "HIGH CONFIDENCE"))
        )

        res_df = pd.DataFrame({
            "sub_id": df_features["sub_id"].values,
            "sub_name": df_features["sub_name"].values if "sub_name" in df_features else df_features["sub_id"].values,
            "lead_day": leads,
            "lead_hour": df_features["lead_hour"].values if "lead_hour" in df_features else leads * 24,
            "confidence_score": conf_scores,
            "bust_probability": b_probs_rounded,
            "alert_level": alert_levels,
            "alert_color": alert_colors,
            "alert_badge": badges,
            "p_clim": np.round(p_clim, 3),
            "p_spread": np.round(p_spread, 3),
            "p_analog": np.round(p_analog, 3),
            "p_gbdt": np.round(p_gbdt, 3),
            "p_spatial": np.round(p_spatial, 3)
        })
        return res_df
