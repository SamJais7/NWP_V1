"""
Project Pratyay (प्रत्यय) - Configuration & Master Constants
Defines domain bounds, 36 IMD subdivisions, 4 homogeneous regions,
IMD operational rainfall thresholds, and benchmark historical cases.
"""

from typing import Dict, List, Any

# ==============================================================================
# 1. SPATIAL DOMAIN SPECIFICATIONS
# ==============================================================================

# Input Domain (Synoptic & Tropical Precursors)
INPUT_DOMAIN = {
    "lon_min": 40.0,
    "lon_max": 120.0,
    "lat_min": -15.0,
    "lat_max": 50.0,
    "resolution": 0.5,  # degrees
}

# Output & Verification Domain (Indian Landmass & Immediate Waters)
TARGET_DOMAIN = {
    "lon_min": 68.0,
    "lon_max": 98.0,
    "lat_min": 5.0,
    "lat_max": 38.0,
    "resolution": 0.25,  # degrees (matches IMD gridded rainfall)
}

# Physical Constants
EARTH_RADIUS = 6371000.0  # meters
GRAVITY = 9.80665         # m/s^2

# ==============================================================================
# 2. LEAD TIMES & TEMPORAL HORIZON
# ==============================================================================

LEAD_DAYS = list(range(1, 11))  # Day 1 through Day 10
LEAD_HOURS = [day * 24 for day in LEAD_DAYS]  # 24h to 240h

# Operational Cycles
CYCLES = ["00Z", "12Z"]

# ==============================================================================
# 3. 36 IMD METEOROLOGICAL SUBDIVISIONS
# ==============================================================================

IMD_SUBDIVISIONS: Dict[str, Dict[str, Any]] = {
    "SUB_01": {"name": "Andaman & Nicobar Islands", "abbr": "ANI", "region": "ENE", "lat": 11.5, "lon": 92.7},
    "SUB_02": {"name": "Arunachal Pradesh", "abbr": "ARU", "region": "ENE", "lat": 28.2, "lon": 94.7},
    "SUB_03": {"name": "Assam & Meghalaya", "abbr": "ASM", "region": "ENE", "lat": 26.2, "lon": 92.5},
    "SUB_04": {"name": "Nagaland, Manipur, Mizoram & Tripura", "abbr": "NMMT", "region": "ENE", "lat": 24.5, "lon": 93.5},
    "SUB_05": {"name": "Sub-Himalayan West Bengal & Sikkim", "abbr": "SHWB", "region": "ENE", "lat": 26.8, "lon": 88.4},
    "SUB_06": {"name": "Gangetic West Bengal", "abbr": "GWB", "region": "ENE", "lat": 23.2, "lon": 87.8},
    "SUB_07": {"name": "Odisha", "abbr": "ODI", "region": "CNI", "lat": 20.5, "lon": 84.4},
    "SUB_08": {"name": "Jharkhand", "abbr": "JHK", "region": "ENE", "lat": 23.6, "lon": 85.3},
    "SUB_09": {"name": "Bihar", "abbr": "BIH", "region": "ENE", "lat": 25.6, "lon": 85.8},
    "SUB_10": {"name": "East Uttar Pradesh", "abbr": "EUP", "region": "NWI", "lat": 26.8, "lon": 82.2},
    "SUB_11": {"name": "West Uttar Pradesh", "abbr": "WUP", "region": "NWI", "lat": 28.0, "lon": 78.5},
    "SUB_12": {"name": "Uttarakhand", "abbr": "UTK", "region": "NWI", "lat": 30.1, "lon": 79.2},
    "SUB_13": {"name": "Haryana, Chandigarh & Delhi", "abbr": "HCD", "region": "NWI", "lat": 29.1, "lon": 76.8},
    "SUB_14": {"name": "Punjab", "abbr": "PUN", "region": "NWI", "lat": 31.0, "lon": 75.4},
    "SUB_15": {"name": "Himachal Pradesh", "abbr": "HP", "region": "NWI", "lat": 31.8, "lon": 77.2},
    "SUB_16": {"name": "Jammu & Kashmir and Ladakh", "abbr": "JKL", "region": "NWI", "lat": 34.0, "lon": 76.0},
    "SUB_17": {"name": "West Rajasthan", "abbr": "WRA", "region": "NWI", "lat": 26.8, "lon": 72.0},
    "SUB_18": {"name": "East Rajasthan", "abbr": "ERA", "region": "NWI", "lat": 25.8, "lon": 75.8},
    "SUB_19": {"name": "West Madhya Pradesh", "abbr": "WMP", "region": "CNI", "lat": 23.2, "lon": 76.5},
    "SUB_20": {"name": "East Madhya Pradesh", "abbr": "EMP", "region": "CNI", "lat": 23.5, "lon": 80.8},
    "SUB_21": {"name": "Gujarat Region", "abbr": "GUJ", "region": "CNI", "lat": 22.5, "lon": 72.8},
    "SUB_22": {"name": "Saurashtra & Kutch", "abbr": "SAU", "region": "CNI", "lat": 22.3, "lon": 70.4},
    "SUB_23": {"name": "Konkan & Goa", "abbr": "KNG", "region": "CNI", "lat": 17.5, "lon": 73.5},
    "SUB_24": {"name": "Madhya Maharashtra", "abbr": "MMH", "region": "CNI", "lat": 18.2, "lon": 74.8},
    "SUB_25": {"name": "Marathwada", "abbr": "MAR", "region": "CNI", "lat": 19.1, "lon": 76.4},
    "SUB_26": {"name": "Vidarbha", "abbr": "VID", "region": "CNI", "lat": 21.0, "lon": 79.3},
    "SUB_27": {"name": "Chhattisgarh", "abbr": "CHH", "region": "CNI", "lat": 21.3, "lon": 81.8},
    "SUB_28": {"name": "Coastal Andhra Pradesh & Yanam", "abbr": "CAP", "region": "SPI", "lat": 16.5, "lon": 81.5},
    "SUB_29": {"name": "Telangana", "abbr": "TEL", "region": "SPI", "lat": 17.9, "lon": 79.1},
    "SUB_30": {"name": "Rayalaseema", "abbr": "RAY", "region": "SPI", "lat": 14.5, "lon": 78.5},
    "SUB_31": {"name": "Tamil Nadu, Puducherry & Karaikal", "abbr": "TNP", "region": "SPI", "lat": 11.1, "lon": 78.6},
    "SUB_32": {"name": "Coastal Karnataka", "abbr": "CKA", "region": "SPI", "lat": 14.0, "lon": 74.5},
    "SUB_33": {"name": "North Interior Karnataka", "abbr": "NIK", "region": "SPI", "lat": 16.0, "lon": 75.8},
    "SUB_34": {"name": "South Interior Karnataka", "abbr": "SIK", "region": "SPI", "lat": 13.0, "lon": 76.5},
    "SUB_35": {"name": "Kerala & Mahe", "abbr": "KER", "region": "SPI", "lat": 10.2, "lon": 76.4},
    "SUB_36": {"name": "Lakshadweep", "abbr": "LAK", "region": "SPI", "lat": 10.5, "lon": 72.6},
}

# ==============================================================================
# 4. 4 HOMOGENEOUS MONSOON REGIONS
# ==============================================================================

HOMOGENEOUS_REGIONS: Dict[str, Dict[str, Any]] = {
    "NWI": {
        "name": "Northwest India",
        "subdivisions": ["SUB_10", "SUB_11", "SUB_12", "SUB_13", "SUB_14", "SUB_15", "SUB_16", "SUB_17", "SUB_18"],
        "color": "#38bdf8"
    },
    "CNI": {
        "name": "Central India",
        "subdivisions": ["SUB_07", "SUB_19", "SUB_20", "SUB_21", "SUB_22", "SUB_23", "SUB_24", "SUB_25", "SUB_26", "SUB_27"],
        "color": "#fb923c"
    },
    "SPI": {
        "name": "South Peninsular India",
        "subdivisions": ["SUB_28", "SUB_29", "SUB_30", "SUB_31", "SUB_32", "SUB_33", "SUB_34", "SUB_35", "SUB_36"],
        "color": "#34d399"
    },
    "ENE": {
        "name": "East & Northeast India",
        "subdivisions": ["SUB_01", "SUB_02", "SUB_03", "SUB_04", "SUB_05", "SUB_06", "SUB_08", "SUB_09"],
        "color": "#a78bfa"
    }
}

# ==============================================================================
# 5. IMD OPERATIONAL RAINFALL CRITERIA (mm/day)
# ==============================================================================

IMD_RAIN_THRESHOLDS = {
    "VERY_LIGHT": (0.1, 2.4),
    "LIGHT": (2.5, 15.5),
    "MODERATE": (15.6, 64.4),
    "HEAVY": (64.5, 115.5),
    "VERY_HEAVY": (115.6, 204.4),
    "EXTREMELY_HEAVY": (204.5, 9999.0)
}

# Bust Definitions
BUST_CRITERIA = {
    "HEAVY_RAIN_MISS": {"obs_min": 64.5, "fcst_max": 35.5},
    "FALSE_ALARM": {"fcst_min": 64.5, "obs_max": 15.5},
    "EXTREME_MISS": {"obs_min": 115.6, "ratio_max": 0.5},
    "P90_STANDARD": 0.90
}

# Alert Categories & Color Thresholds
ALERT_LEVELS = {
    "HIGH": {"label": "HIGH CONFIDENCE", "prob_max": 0.25, "conf_min": 75.0, "color": "#10b981", "bg": "bg-emerald-950"},
    "MODERATE": {"label": "MODERATE WATCH", "prob_max": 0.50, "conf_min": 50.0, "color": "#eab308", "bg": "bg-yellow-950"},
    "LOW": {"label": "LOW CONFIDENCE", "prob_max": 0.75, "conf_min": 25.0, "color": "#f97316", "bg": "bg-amber-950"},
    "CRITICAL": {"label": "CRITICAL BUST WARNING", "prob_max": 1.0, "conf_min": 0.0, "color": "#f43f5e", "bg": "bg-rose-950"}
}

# ==============================================================================
# 6. HELD-OUT BENCHMARK CASE STUDIES
# ==============================================================================

BENCHMARK_CASES = {
    "kerala_2018": {
        "title": "Kerala Extreme Rain & Floods",
        "dates": "8–16 August 2018",
        "primary_subdivision": "SUB_35",
        "key_synoptic": "Somali LLJ Surge + Offshore Trough with deep BoB low",
        "failure_mode": "Model underestimated multi-day persistent orographic deluge by >120 mm/day at Day 3–5",
        "observed_peak_rain": 242.0,
        "raw_forecast_rain": 48.0,
        "ground_truth_bust": True,
        "regime": "SOMALI_JET_SURGE"
    },
    "amphan_2020": {
        "title": "Super Cyclone Amphan",
        "dates": "16–21 May 2020",
        "primary_subdivision": "SUB_06",
        "key_synoptic": "Rapid Intensification over warm Bay of Bengal SST (>31°C)",
        "failure_mode": "Under-predicted peak coastal landfall squalls & inland penetration track curve",
        "observed_peak_rain": 185.0,
        "raw_forecast_rain": 52.0,
        "ground_truth_bust": True,
        "regime": "BAY_OF_BENGAL_DEPRESSION"
    },
    "biparjoy_2023": {
        "title": "Very Severe Cyclone Biparjoy",
        "dates": "6–17 June 2023",
        "primary_subdivision": "SUB_22",
        "key_synoptic": "Erratic Arabian Sea recurvature & interaction with mid-latitude trough",
        "failure_mode": "Consecutive run jumpiness (>180 km track displacement) across 48h lead times",
        "observed_peak_rain": 140.0,
        "raw_forecast_rain": 30.0,
        "ground_truth_bust": True,
        "regime": "ARABIAN_SEA_CYCLONE"
    },
    "break_aug2023": {
        "title": "Extended Monsoon Break Spell",
        "dates": "5–18 August 2023",
        "primary_subdivision": "SUB_20",
        "key_synoptic": "Monsoon trough shifted to Himalayan foothills; LLJ weakened",
        "failure_mode": "Model repeatedly predicted premature revival of central Indian rainfall at Day 4–7",
        "observed_peak_rain": 2.5,
        "raw_forecast_rain": 68.0,
        "ground_truth_bust": True,
        "regime": "MONSOON_BREAK"
    }
}
