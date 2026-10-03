"""
Historical Precedent Analogs Matcher.
Retrieves documented historical forecast failure cases matching the current synoptic state.
"""

from typing import Dict, List, Any
import numpy as np
from src.config import BENCHMARK_CASES


class PrecedentAnalogFinder:
    """Matches current atmospheric state to documented high-impact forecast busts."""

    def __init__(self):
        self.benchmarks = BENCHMARK_CASES

    def find_matching_analogs(
        self,
        sub_id: str,
        regime: str,
        features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Computes similarity between current state and historical benchmark busts.
        Returns top 2 matching analog cards.
        """
        matches = []
        llj_cur = features.get("somali_llj_index", 12.0)
        apcp_cur = features.get("apcp_mean", 15.0)

        for case_id, case in self.benchmarks.items():
            sim = 0.50  # base match

            # Regime alignment
            if case["regime"] == regime:
                sim += 0.30

            # Sub-division alignment
            if case["primary_subdivision"] == sub_id:
                sim += 0.15

            # Flow similarity
            if case_id == "kerala_2018" and sub_id in ["SUB_35", "SUB_32", "SUB_23"]:
                sim = max(sim, 0.95 if llj_cur > 15.0 else 0.82)
            elif case_id == "amphan_2020" and sub_id in ["SUB_06", "SUB_07"]:
                sim = max(sim, 0.92 if apcp_cur > 30.0 else 0.78)
            elif case_id == "biparjoy_2023" and sub_id in ["SUB_22", "SUB_21", "SUB_17"]:
                sim = max(sim, 0.91)
            elif case_id == "break_aug2023" and regime == "MONSOON_BREAK":
                sim = max(sim, 0.88)

            match_pct = round(min(0.99, sim) * 100, 1)

            matches.append({
                "case_id": case_id,
                "title": case["title"],
                "dates": case["dates"],
                "similarity": match_pct,
                "key_synoptic": case["key_synoptic"],
                "failure_mode": case["failure_mode"],
                "observed_peak_rain": case["observed_peak_rain"],
                "raw_forecast_rain": case["raw_forecast_rain"]
            })

        # Sort by similarity
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches[:2]
