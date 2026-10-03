"""
Pydantic v2 Schemas for Operational FastAPI Endpoints.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class ModelLadderScores(BaseModel):
    climatology_prob: float
    spread_prob: float
    analog_prob: float
    gbdt_prob: float
    spatial_prob: float


class DriverItem(BaseModel):
    feature: str
    shap_value: float
    direction: str
    unit: str
    observed_value: float
    impact: str


class AnalogItem(BaseModel):
    case_id: str
    title: str
    dates: str
    similarity: float
    key_synoptic: str
    failure_mode: str
    observed_peak_rain: float
    raw_forecast_rain: float


class AdvisoryBulletin(BaseModel):
    section_1_regime: str
    section_2_drivers: str
    section_3_analogs: str
    section_4_recommendation: str
    full_bulletin: str


class SubdivisionAssessment(BaseModel):
    sub_id: str
    sub_name: str
    region: str
    lead_day: int
    lead_hour: int
    confidence_score: float
    bust_probability: float
    alert_level: str
    alert_color: str
    alert_badge: str
    model_ladder: ModelLadderScores
    regime_info: Dict[str, Any]
    top_drivers: List[DriverItem]
    analogs: List[AnalogItem]
    bulletin: AdvisoryBulletin


class TrajectoryPoint(BaseModel):
    lead_day: int
    lead_hour: int
    confidence_score: float
    bust_probability: float
    alert_level: str
    alert_color: str


class HomogeneousSummaryItem(BaseModel):
    name: str
    color: str
    mean_confidence_day1: float


class ConfidenceResponse(BaseModel):
    metadata: Dict[str, Any]
    homogeneous_summary: Dict[str, HomogeneousSummaryItem]
    lead_day: int
    subdivisions: Dict[str, SubdivisionAssessment]


class SubdivisionDetailResponse(BaseModel):
    sub_id: str
    sub_name: str
    region: str
    current_assessment: SubdivisionAssessment
    trajectory: List[TrajectoryPoint]


class AlertItem(BaseModel):
    sub_id: str
    sub_name: str
    region: str
    lead_day: int
    alert_level: str
    alert_badge: str
    confidence_score: float
    bust_probability: float
    key_reason: str


class AlertsResponse(BaseModel):
    total_active_alerts: int
    critical_alerts_count: int
    low_confidence_count: int
    alerts: List[AlertItem]


class VerificationScorecard(BaseModel):
    evaluation_dataset: str
    bss_vs_climatology: float
    bss_vs_spread: float
    roc_auc: float
    pr_auc: float
    critical_success_index: float
    false_alarm_ratio: float
    hit_rate: float
    spatial_fractions_skill_score: float
    max_calibration_error: float
    ladder_ablation_summary: List[Dict[str, Any]]
