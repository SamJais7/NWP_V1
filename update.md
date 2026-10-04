# Project Pratyay (प्रत्यय) / Satark-Mausam
## Engineering & Operational Upgrade Specification: Real-World Data & Production GIS
**Document:** `update.md`  
**Target Platform:** MoES / NCMRWF / IMD Operational Decision Support System  
**Status:** Approved for Implementation

---

## 1. Executive Summary & Objective

This document outlines the architectural and engineering migrations required to transition Project Pratyay from synthetic/reanalysis-based demonstrations to an **end-to-end real-world operational system**. 

The two primary pillars of this upgrade are:
1. **Model Training & Verification on Live Ground Truth**: Transitioning from pure static archives to live streaming Automatic Weather Station (AWS) data, gridded observations, and operational alerts through official IMD APIs coupled with global NWP forecast archives (Open-Meteo / ECMWF IFS / NOAA GFS).
2. **Production GIS & Cartographic Realism**: Replacing placeholder tile endpoints with **Geoapify Vector & Raster Tiles**, high-precision geocoding, reverse geocoding, and synchronized multi-level administrative boundaries (36 IMD Subdivisions, 700+ Districts, and individual AWS station coordinates).

---

## 2. Component Migration Matrix

| Subsystem | Previous State | Target Production State |
|---|---|---|
| **Ground Truth Ingestion** | Static ERA5 / IMD 0.25° historical NetCDFs | Real-time IMD API suite (`aws_data`, `districtrainfall`, `staterainfall`, `current_wx`) |
| **Operational Forecasts** | Simulated/cached forecast tensors | Programmatic NWP ingestion via Open-Meteo API (ECMWF IFS 9 km, NOAA GFS 0.25°, ECMWF AIFS) |
| **Operational Benchmarks** | Theoretical climatological base rates | Official IMD forecast & hazard feeds (`cityforecastloc`, `districtwarning`, `subdivisionwarning`) |
| **Map Base Layers** | Generic CartoDB dark tile layer | High-resolution **Geoapify Dark/Matter Vector Tiles** with customizable styles |
| **Spatial Matching** | Approximation via bounding boxes | **Geoapify Geocoding API** + Scipy KDTree for exact coordinate validation and polygon containment |
| **Feature Store & Registry** | Hardcoded subdivision centroid list | Station & District Registry DB with dynamic lat/lon lookup and elevation metadata |

---

## 3. Data Architecture & Pipeline Upgrades

### 3.1 IMD Official API Ingestion Hierarchy

The IMD API catalog is structured into three functional pipelines:

```mermaid
flowchart TD
    subgraph INGEST_IMD["IMD Real-Time API Streams"]
        OBS["1. Ground-Truth Stream (Target Variable y)\n• aws_data & aws_data_mapping\n• districtrainfall & staterainfall\n• current_wx (Synop Surface)"]
        FCST["2. Official IMD Baselines\n• cityforecast & cityforecastloc (7-Day)\n• subdivision_rainfall_forecast (7-Day)\n• basinqpf (River Basin QPF)"]
        WARN["3. Hazard Warning Benchmarks\n• districtwarning & subdivisionwarning\n• districtnowcast & stationnowcast (0-3h)\n• coastalbulletin, portwarning, seabulletin"]
    end

    subgraph GEO["Spatial Intelligence Layer"]
        GEOAPIFY["Geoapify APIs\n• Geocoding (Validate Station Names & Lat/Lon)\n• Reverse Geocoding (Resolve GPS clicks to District/Subdivision)\n• Vector Tiles (High-contrast Forecaster Cartography)"]
    end

    subgraph NWP["NWP Model Ingestion Engine"]
        OPEN_METEO["Open-Meteo / NOAA / ECMWF APIs\n• ECMWF IFS 0.25° / GFS 0.25° / AIFS\n• Previous Runs API (Historical forecasts at lead tau = 1..10)"]
    end

    subgraph STORAGE["Feature Store & Alignment Engine"]
        REGISTRY["Station & Geometry Registry\n(station_id, district_id, sub_id, lat, lon, elevation)"]
        PAIRING["Spatial-Temporal Pairing Engine\nMatches Forecast F(x, t, tau) with Observed Truth O(x, t)"]
    end

    OBS --> REGISTRY
    GEOAPIFY --> REGISTRY
    FCST --> PAIRING
    NWP --> PAIRING
    REGISTRY --> PAIRING
```

---

## 4. Configuration & Environment Specifications

```python
# Real-World API Configuration
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY", "")
GEOAPIFY_TILE_STYLE = "dark-matter-dark-grey"

IMD_API_BASE_URL = os.getenv("IMD_API_BASE_URL", "https://mausam.imd.gov.in/api")
IMD_API_KEY = os.getenv("IMD_API_KEY", "")

NWP_FORECAST_PROVIDER = "open-meteo"
NWP_MODELS = ["ecmwf_ifs025", "gfs_seamless", "ecmwf_aifs025"]

# Real-World Storage Paths
STATION_REGISTRY_PATH = "data/registry/imd_aws_registry.parquet"
REAL_WORLD_FEATURE_STORE = "data/processed/features_real_world.parquet"
```
