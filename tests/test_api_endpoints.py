"""
Automated Test Suite for FastAPI Operational Endpoints.
Verifies all 8 REST endpoints, status codes, payload structures, and response models.
"""

# Optional pytest import
try:
    import pytest
except ImportError:
    pass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_01_health_check():
    """Verify system health, operational state, and metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OPERATIONAL"
    assert data["subdivisions_count"] == 36
    assert "endpoints" in data
    print("Test 01 (Health Check): PASS")


def test_02_dashboard_html():
    """Verify HTML serving of single-page GIS Leaflet dashboard."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    text = response.text
    assert "leaflet" in text.lower()
    assert "chart.js" in text.lower()
    assert "Pratyay" in text
    print("Test 02 (Dashboard HTML): PASS")


def test_03_geojson():
    """Verify GeoJSON FeatureCollection for 36 IMD subdivisions."""
    response = client.get("/api/v1/geojson")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 36
    sample = data["features"][0]
    assert "geometry" in sample
    assert "properties" in sample
    assert "sub_id" in sample["properties"]
    print("Test 03 (GeoJSON Features): PASS")


def test_04_confidence_assessment():
    """Verify All-India Day 1-10 confidence assessment endpoint."""
    response = client.get("/api/v1/confidence?lead_day=3")
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert "homogeneous_summary" in data
    assert data["lead_day"] == 3
    assert len(data["subdivisions"]) == 36
    
    # Check Kerala (SUB_35)
    ker = data["subdivisions"]["SUB_35"]
    assert "confidence_score" in ker
    assert "bust_probability" in ker
    assert "model_ladder" in ker
    assert 0.0 <= ker["confidence_score"] <= 100.0
    print("Test 04 (Confidence Assessment): PASS")


def test_05_subdivision_detail():
    """Verify forecaster deep-dive endpoint with 10-day trajectory and XAI."""
    response = client.get("/api/v1/subdivision/SUB_35?lead_day=3")
    assert response.status_code == 200
    data = response.json()
    assert data["sub_id"] == "SUB_35"
    assert data["sub_name"] == "Kerala & Mahe"
    assert len(data["trajectory"]) == 10
    
    # Check advisory and drivers
    cur = data["current_assessment"]
    assert len(cur["top_drivers"]) > 0
    assert "full_bulletin" in cur["bulletin"]
    print("Test 05 (Subdivision Detail & Trajectory): PASS")


def test_06_operational_alerts():
    """Verify active warning alerts endpoint."""
    response = client.get("/api/v1/alerts?lead_day=5")
    assert response.status_code == 200
    data = response.json()
    assert "total_active_alerts" in data
    assert "alerts" in data
    print("Test 06 (Operational Alerts): PASS")


def test_07_verification_scorecard():
    """Verify official scorecard and ablation ladder rungs."""
    response = client.get("/api/v1/verification")
    assert response.status_code == 200
    data = response.json()
    assert data["bss_vs_climatology"] >= 0.25
    assert data["roc_auc"] >= 0.82
    assert len(data["ladder_ablation_summary"]) == 6
    print("Test 07 (Verification Scorecard & Ablation): PASS")


def test_08_benchmark_scenario():
    """Verify held-out disaster case study endpoint."""
    response = client.get("/api/v1/benchmark/kerala_2018")
    assert response.status_code == 200
    data = response.json()
    assert data["case_id"] == "kerala_2018"
    assert "Kerala Extreme Rain" in data["metadata"]["title"]
    assert data["metadata"]["primary_subdivision"] == "SUB_35"
    print("Test 08 (Benchmark Scenario Kerala 2018): PASS")


def test_09_gis_config():
    """Verify Geoapify GIS tile configuration and styles."""
    response = client.get("/api/v1/config/gis")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "Geoapify"
    assert "tile_url_template" in data
    assert len(data["available_styles"]) >= 3
    print("Test 09 (Geoapify GIS Config): PASS")


def test_10_live_stations():
    """Verify Automatic Weather Station (AWS) live telemetry stream."""
    response = client.get("/api/v1/stations")
    assert response.status_code == 200
    data = response.json()
    assert data["total_stations"] > 0
    st = data["stations"][0]
    assert "live_telemetry" in st
    assert "temp_c" in st["live_telemetry"]
    print("Test 10 (Live AWS Stations Telemetry): PASS")


def test_11_districts_geojson():
    """Verify Level 2 District boundaries GeoJSON with synchronized IMD warnings."""
    response = client.get("/api/v1/districts")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0
    p = data["features"][0]["properties"]
    assert "warning_color" in p
    assert "warning_hex" in p
    print("Test 11 (District Boundaries & Warnings): PASS")


def test_12_geocoding_search():
    """Verify Geoapify Geocoding address search."""
    response = client.get("/api/v1/geocode/search?q=Pune")
    assert response.status_code == 200
    data = response.json()
    assert "latitude" in data
    assert "longitude" in data
    print("Test 12 (Geoapify Geocoding Search): PASS")


def test_13_reverse_geocoding():
    """Verify map click reverse geocoding to District, State, and AWS station."""
    response = client.get("/api/v1/geocode/reverse?lat=18.5204&lon=73.8567")
    assert response.status_code == 200
    data = response.json()
    assert "district" in data
    assert "subdivision_id" in data
    assert "nearest_station" in data
    assert "live_aws_telemetry" in data
    print("Test 13 (Reverse Geocoding & Nearest AWS): PASS")


def test_14_imd_warnings():
    """Verify official IMD warning feeds for side-by-side benchmark."""
    response = client.get("/api/v1/imd/warnings")
    assert response.status_code == 200
    data = response.json()
    assert "subdivision_warnings" in data
    assert "district_warnings" in data
    print("Test 14 (IMD Official Warning Feeds): PASS")


def test_15_real_world_pipeline_status():
    """Verify pipeline status, feature store size, and benchmark scores."""
    response = client.get("/api/v1/real_world/status")
    assert response.status_code == 200
    data = response.json()
    assert data["pipeline_state"] == "ACTIVE_SYNCHRONIZED"
    assert data["features_extracted"] >= 100
    assert data["roc_auc"] >= 0.82
    print("Test 15 (Real-World Pipeline Status & Benchmarks): PASS")


if __name__ == "__main__":
    test_01_health_check()
    test_02_dashboard_html()
    test_03_geojson()
    test_04_confidence_assessment()
    test_05_subdivision_detail()
    test_06_operational_alerts()
    test_07_verification_scorecard()
    test_08_benchmark_scenario()
    test_09_gis_config()
    test_10_live_stations()
    test_11_districts_geojson()
    test_12_geocoding_search()
    test_13_reverse_geocoding()
    test_14_imd_warnings()
    test_15_real_world_pipeline_status()
    print("\n[PASS] ALL 15 OPERATIONAL & REAL-WORLD API TESTS PASSED WITH 100% SUCCESS!")
