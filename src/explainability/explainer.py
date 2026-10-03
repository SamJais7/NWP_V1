"""
Context-Aware Feature Attribution & Indian Synoptic Pattern Classifier.
Translates mathematical model activations into physically grounded meteorological drivers.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd


class SynopticPatternClassifier:
    """Classifies the prevailing Indian synoptic regime based on atmospheric field indices."""

    @staticmethod
    def classify_regime(features: Dict[str, float]) -> Dict[str, Any]:
        """
        Diagnoses active synoptic patterns:
          - SOMALI_JET_SURGE: Strong Arabian Sea LLJ (> 16 m/s) with high moisture
          - BAY_OF_BENGAL_DEPRESSION: High 850 hPa vorticity / deep MSLP anomaly in east
          - WESTERN_DISTURBANCE: Deep 500 hPa trough over NW India (wd_depth < -25 gpm)
          - MONSOON_BREAK: Monsoon trough shifted north (> 27°N) or weak central winds
          - PRE_MONSOON_HEATWAVE: High T2m (> 315 K / 42°C) with dry air
          - ACTIVE_MONSOON: Well-positioned trough (20°–24°N) with strong westerly flow
        """
        llj = features.get("somali_llj_index", 10.0)
        trough_lat = features.get("monsoon_trough_lat", 22.0)
        wd_depth = features.get("wd_depth", 0.0)
        apcp = features.get("apcp_mean", 10.0)
        t2m = features.get("t2m_mean", 300.0)
        mfc = features.get("mfc_mean", 0.0)

        tags = []
        primary_regime = "ACTIVE_MONSOON"
        regime_desc = "Standard South-West Monsoon circulation across Central India."

        if wd_depth < -25.0:
            tags.append("Western Disturbance Trough")
            primary_regime = "WESTERN_DISTURBANCE"
            regime_desc = "Mid-latitude upper-tropospheric westerly trough propagating across NW India."
        
        if llj >= 16.5:
            tags.append("Somali LLJ Surge")
            primary_regime = "SOMALI_JET_SURGE"
            regime_desc = "Intensified Low-Level Jet (LLJ > 16.5 m/s) driving heavy Western Ghats moisture flux."

        if trough_lat >= 26.5 and apcp < 8.0:
            tags.append("Monsoon Break Phase")
            primary_regime = "MONSOON_BREAK"
            regime_desc = "Monsoon trough axis displaced north toward the Himalayan foothills; dry spell across central plains."

        if apcp >= 35.0 or (features.get("zeta850_mean", 0.0) > 4.0):
            tags.append("Bay of Bengal Depression / Low")
            primary_regime = "BAY_OF_BENGAL_DEPRESSION"
            regime_desc = "Low-pressure system / Monsoon Depression inducing concentrated cyclonic convergence."

        if t2m >= 315.15:  # > 42°C
            tags.append("Indo-Gangetic Heatwave")
            primary_regime = "PRE_MONSOON_HEATWAVE"
            regime_desc = "Subsidence-induced extreme surface heating over the Indo-Gangetic plains."

        if not tags:
            tags.append("Active SW Monsoon")

        return {
            "primary_regime": primary_regime,
            "regime_title": tags[0],
            "description": regime_desc,
            "all_tags": tags
        }


class ContextAwareExplainer:
    """Computes directional SHAP-style attribution for duty forecasters."""

    def __init__(self):
        self.classifier = SynopticPatternClassifier()

    def attribute_subdivision_risk(
        self,
        features: Dict[str, float],
        bust_prob: float
    ) -> List[Dict[str, Any]]:
        """
        Attributes the top physical drivers contributing to or mitigating bust risk.
        Returns sorted list of drivers with physical name, SHAP value (+/-), and explanation.
        """
        drivers = []
        
        # 1. Somali LLJ Factor
        llj = features.get("somali_llj_index", 10.0)
        if llj > 15.0:
            shap_val = 0.12 * (llj - 14.0)
            drivers.append({
                "feature": "Somali LLJ Speed",
                "shap_value": round(shap_val, 3),
                "direction": "INCREASES_RISK",
                "unit": "m/s",
                "observed_value": round(llj, 1),
                "impact": f"High low-level jet speed ({llj:.1f} m/s) creates strong cross-equatorial surge and orographic deluges."
            })
        elif llj < 8.0:
            shap_val = 0.08 * (10.0 - llj)
            drivers.append({
                "feature": "Weak Monsoon Circulation",
                "shap_value": round(shap_val, 3),
                "direction": "INCREASES_RISK",
                "unit": "m/s",
                "observed_value": round(llj, 1),
                "impact": f"Sluggish Somali jet ({llj:.1f} m/s) impairs model ability to predict revival timing."
            })

        # 2. Ensemble Spread Deficit
        spread = features.get("apcp_spread_mean", 4.0)
        lead_d = features.get("lead_day", 3)
        expected_spread = 2.0 + 1.2 * lead_d
        if spread > expected_spread * 1.25:
            shap_val = 0.18 * (spread / expected_spread - 1.0)
            drivers.append({
                "feature": "High Ensemble Divergence",
                "shap_value": round(shap_val, 3),
                "direction": "INCREASES_RISK",
                "unit": "mm/day",
                "observed_value": round(spread, 1),
                "impact": f"5-member GEFS spread ({spread:.1f} mm) exceeds climatological baseline by {((spread/expected_spread)-1)*100:.0f}%."
            })

        # 3. Forecast Jumpiness / Tendency
        jump = features.get("jump_apcp", 1.5)
        if jump > 3.0:
            shap_val = 0.15 * (jump / 3.0)
            drivers.append({
                "feature": "Run-to-Run Jumpiness (ΔF_12h)",
                "shap_value": round(shap_val, 3),
                "direction": "INCREASES_RISK",
                "unit": "mm/day",
                "observed_value": round(jump, 1),
                "impact": f"Previous 12h cycle differed by {jump:.1f} mm for the same valid verification time."
            })

        # 4. Vertical Wind Shear
        shear = features.get("shear_mean", 15.0)
        if shear > 25.0:
            shap_val = 0.10 * ((shear - 20.0) / 10.0)
            drivers.append({
                "feature": "Deep Vertical Wind Shear",
                "shap_value": round(shap_val, 3),
                "direction": "INCREASES_RISK",
                "unit": "m/s",
                "observed_value": round(shear, 1),
                "impact": f"Strong vertical shear ({shear:.1f} m/s) disrupts organized deep convection tracks."
            })
        elif shear < 10.0:
            drivers.append({
                "feature": "Weak Wind Shear",
                "shap_value": -0.06,
                "direction": "DECREASES_RISK",
                "unit": "m/s",
                "observed_value": round(shear, 1),
                "impact": "Favorable environmental shear enhances deterministic forecast skill."
            })

        # 5. Moisture Flux / PWAT
        pwat = features.get("pwat_mean", 50.0)
        if pwat > 60.0:
            shap_val = 0.11 * ((pwat - 55.0) / 10.0)
            drivers.append({
                "feature": "Atmospheric River / PWAT Anomaly",
                "shap_value": round(shap_val, 3),
                "direction": "INCREASES_RISK",
                "unit": "kg/m²",
                "observed_value": round(pwat, 1),
                "impact": f"Extreme precipitable water ({pwat:.1f} kg/m²) elevates localized flash flooding probability."
            })

        # Sort by absolute SHAP impact
        drivers.sort(key=lambda d: abs(d["shap_value"]), reverse=True)
        return drivers[:4]
