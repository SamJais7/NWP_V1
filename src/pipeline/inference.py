"""
End-to-End Operational Inference Orchestrator for Project Pratyay.
Executes the progressive model ladder, probability calibration, XAI feature attribution,
and advisory synthesis across all 36 subdivisions for Day 1 through Day 10.
"""

from typing import Dict, List, Any, Optional
import datetime
import numpy as np
import pandas as pd
import xarray as xr

from src.config import IMD_SUBDIVISIONS, HOMOGENEOUS_REGIONS, LEAD_DAYS, LEAD_HOURS, BENCHMARK_CASES
from src.data.fetcher import DataFetcher
from src.data.preprocessor import AtmosphericDynamicsPreprocessor
from src.data.regions import SubdivisionManager
from src.data.feature_engineer import FeatureEngineer
from src.models.spatial_model import SpatialBustUNet
from src.models.ensemble import MetaEnsembleStacker
from src.explainability.explainer import ContextAwareExplainer, SynopticPatternClassifier
from src.explainability.analog_finder import PrecedentAnalogFinder
from src.explainability.templates import AdvisoryGenerator


class InferenceOrchestrator:
    """Coordinates data fetch, dynamic transforms, model stacking, XAI, and advisory creation."""

    def __init__(self, offline_mode: bool = True):
        self.fetcher = DataFetcher(offline_mode=offline_mode)
        self.preprocessor = AtmosphericDynamicsPreprocessor()
        self.region_manager = SubdivisionManager()
        self.feature_engineer = FeatureEngineer()
        self.spatial_model = SpatialBustUNet(in_channels=15)
        self.stacker = MetaEnsembleStacker()
        self.explainer = ContextAwareExplainer()
        self.regime_classifier = SynopticPatternClassifier()
        self.analog_finder = PrecedentAnalogFinder()
        self.advisory_gen = AdvisoryGenerator()

    def run_cycle_assessment(
        self,
        init_date: str = "2026-07-15",
        cycle: str = "00Z",
        scenario: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs the full assessment pipeline for 36 subdivisions and 10 lead days.
        """
        # 1. Fetch raw forecast data
        ds_raw = self.fetcher.fetch_forecast_cycle(init_date, cycle, scenario=scenario)

        # 2. Compute atmospheric dynamics
        ds_enriched = self.preprocessor.process_dataset(ds_raw)

        # 3. Extract tabular features
        df_features = self.feature_engineer.extract_features_for_cycle(ds_enriched)

        # 4. Meta-Ensemble Stacking & Probability Calibration
        df_predictions = self.stacker.predict(df_features)

        # 5. Build lookup structures
        # Structure by lead_day: { lead_day: { sub_id: {...} } }
        leads_data: Dict[int, Dict[str, Any]] = {d: {} for d in LEAD_DAYS}
        subdivisions_trajectory: Dict[str, List[Dict[str, Any]]] = {s: [] for s in IMD_SUBDIVISIONS}

        # Merge predictions with features for XAI
        merged = df_predictions.merge(df_features, on=["sub_id", "sub_name", "lead_day", "lead_hour"])

        for _, row in merged.iterrows():
            sub_id = row["sub_id"]
            lead_d = int(row["lead_day"])
            row_dict = row.to_dict()

            # Regime
            regime_info = self.regime_classifier.classify_regime(row_dict)

            # XAI SHAP Drivers
            drivers = self.explainer.attribute_subdivision_risk(row_dict, row["bust_probability"])

            # Analogs
            analogs = self.analog_finder.find_matching_analogs(sub_id, regime_info["primary_regime"], row_dict)

            # Operational Bulletin
            bulletin = self.advisory_gen.generate_advisory(
                sub_name=row["sub_name"],
                lead_day=lead_d,
                confidence_score=row["confidence_score"],
                bust_prob=row["bust_probability"],
                regime_info=regime_info,
                top_drivers=drivers,
                analogs=analogs
            )

            item = {
                "sub_id": sub_id,
                "sub_name": row["sub_name"],
                "region": IMD_SUBDIVISIONS[sub_id]["region"],
                "lead_day": lead_d,
                "lead_hour": int(row["lead_hour"]),
                "confidence_score": float(row["confidence_score"]),
                "bust_probability": float(row["bust_probability"]),
                "alert_level": row["alert_level"],
                "alert_color": row["alert_color"],
                "alert_badge": row["alert_badge"],
                "model_ladder": {
                    "climatology_prob": float(row["p_clim"]),
                    "spread_prob": float(row["p_spread"]),
                    "analog_prob": float(row["p_analog"]),
                    "gbdt_prob": float(row["p_gbdt"]),
                    "spatial_prob": float(row["p_spatial"])
                },
                "regime_info": regime_info,
                "top_drivers": drivers,
                "analogs": analogs,
                "bulletin": bulletin
            }

            leads_data[lead_d][sub_id] = item
            subdivisions_trajectory[sub_id].append({
                "lead_day": lead_d,
                "lead_hour": int(row["lead_hour"]),
                "confidence_score": float(row["confidence_score"]),
                "bust_probability": float(row["bust_probability"]),
                "alert_level": row["alert_level"],
                "alert_color": row["alert_color"]
            })

        # Calculate Regional Homogeneous Aggregates
        homogeneous_summary = {}
        for reg_code, reg_info in HOMOGENEOUS_REGIONS.items():
            subs = reg_info["subdivisions"]
            reg_confs = [leads_data[1][s]["confidence_score"] for s in subs if s in leads_data[1]]
            homogeneous_summary[reg_code] = {
                "name": reg_info["name"],
                "color": reg_info["color"],
                "mean_confidence_day1": round(float(np.mean(reg_confs)), 1) if reg_confs else 85.0
            }

        return {
            "metadata": {
                "init_date": init_date,
                "cycle": cycle,
                "scenario": scenario or "Operational Forecast",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model_platform": "NOAA-GEFSv12 / NCUM-G Standardized"
            },
            "homogeneous_summary": homogeneous_summary,
            "leads": leads_data,
            "trajectories": subdivisions_trajectory
        }
