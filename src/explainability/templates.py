"""
Natural Language Operational Advisory Synthesis Engine.
Formats structured 4-section MoES/IMD bulletins for duty forecasters.
"""

from typing import Dict, List, Any


class AdvisoryGenerator:
    """Generates standardized 4-part operational advisories for meteorological forecasters."""

    @staticmethod
    def generate_advisory(
        sub_name: str,
        lead_day: int,
        confidence_score: float,
        bust_prob: float,
        regime_info: Dict[str, Any],
        top_drivers: List[Dict[str, Any]],
        analogs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes a 4-section bulletin.
        """
        # Section 1: Regime
        regime_title = regime_info.get("regime_title", "Active SW Monsoon")
        regime_desc = regime_info.get("description", "Normal monsoon circulation.")
        sec1 = f"PREVAILING REGIME: {regime_title.upper()}.\n{regime_desc}"

        # Section 2: Physical Drivers
        sec2_lines = []
        for d in top_drivers:
            impact_sign = "+" if d["shap_value"] > 0 else "-"
            sec2_lines.append(
                f"• {d['feature']} ({d['observed_value']} {d.get('unit', '')}): "
                f"SHAP {impact_sign}{abs(d['shap_value']):.2f} — {d['impact']}"
            )
        sec2 = "\n".join(sec2_lines) if sec2_lines else "• Environmental parameters within normal climatological bounds."

        # Section 3: Analogs
        sec3_lines = []
        for a in analogs:
            sec3_lines.append(
                f"• {a['title']} ({a['dates']}) [Similarity: {a['similarity']}%]:\n"
                f"  Failure Mode: {a['failure_mode']}.\n"
                f"  Observed Rain: {a['observed_peak_rain']} mm/day vs Forecast: {a['raw_forecast_rain']} mm/day."
            )
        sec3 = "\n\n".join(sec3_lines) if sec3_lines else "• No severe historical failure analogs detected."

        # Section 4: Operational Recommendation
        if bust_prob >= 0.75 or confidence_score < 25.0:
            action = (
                f"CRITICAL WARNING (Confidence: {confidence_score:.1f}%): "
                f"High probability of severe forecast divergence at Day {lead_day}. "
                "Duty forecaster is advised to avoid single-model deterministic reliance. "
                "Cross-check high-resolution radar/satellite nowcasts, consult ensemble plume spread, "
                "and consider issuing precautionary Orange/Red district warnings for localized deluges."
            )
        elif bust_prob >= 0.50 or confidence_score < 50.0:
            action = (
                f"ELEVATED RISK WATCH (Confidence: {confidence_score:.1f}%): "
                f"Moderate-to-high forecast uncertainty at Day {lead_day}. "
                "Model shows tendency jumpiness. Forecasters should apply dampening factor to extreme values "
                "and monitor next 12Z cycle update."
            )
        elif bust_prob >= 0.25 or confidence_score < 75.0:
            action = (
                f"MODERATE CONFIDENCE (Confidence: {confidence_score:.1f}%): "
                f"Forecast acceptable for general synoptic guidance at Day {lead_day}. "
                "Verify orographic boundary layer convergence along coastal belts."
            )
        else:
            action = (
                f"HIGH CONFIDENCE (Confidence: {confidence_score:.1f}%): "
                f"NWP deterministic guidance is highly reliable at Day {lead_day}. "
                "Operational alerts may proceed with standard confidence margins."
            )

        full_bulletin = (
            f"=== MOES / NCMRWF FORECAST CONFIDENCE BULLETIN ===\n"
            f"SUBDIVISION: {sub_name.upper()} | LEAD: DAY {lead_day} ({lead_day * 24}H)\n"
            f"CALIBRATED CONFIDENCE: {confidence_score:.1f}% | BUST PROBABILITY: {bust_prob:.1%}\n\n"
            f"[SECTION 1: SYNOPTIC REGIME DIAGNOSTIC]\n{sec1}\n\n"
            f"[SECTION 2: KEY PHYSICAL DRIVERS (SHAP ATTRIBUTION)]\n{sec2}\n\n"
            f"[SECTION 3: HISTORICAL PRECEDENT ANALOGS]\n{sec3}\n\n"
            f"[SECTION 4: OPERATIONAL RECOMMENDATION FOR DUTY FORECASTER]\n{action}"
        )

        return {
            "section_1_regime": sec1,
            "section_2_drivers": sec2,
            "section_3_analogs": sec3,
            "section_4_recommendation": action,
            "full_bulletin": full_bulletin
        }
