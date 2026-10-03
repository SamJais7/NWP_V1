# PART I: Product Requirements Document (PRD)

## 1. Document & Project Metadata

| Attribute | Specification |
|---|---|
| **Project Code Name** | **Pratyay (प्रत्यय)** / Satark-Mausam |
| **Problem Statement ID** | 26079 |
| **Problem Statement Title** | AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts |
| **Nodal Ministry / Dept** | Ministry of Earth Sciences (MoES) / NCMRWF & IMD |
| **Target End-Users** | Operational Duty Forecasters (IMD National/Regional Centers), NCMRWF Model Diagnostic Scientists, State Disaster Management Authorities (SDMAs) |
| **Domain Scope** | **Input:** Synoptic/Tropical Context ($40^\circ\text{E}–120^\circ\text{E}, 15^\circ\text{S}–50^\circ\text{N}$)<br>**Output & Verification:** Indian Subcontinent ($5^\circ\text{N}–38^\circ\text{N}, 68^\circ\text{E}–98^\circ\text{E}$; 36 IMD Subdivisions & Coastal Oceanic Zones) |
| **Target Operational Cycle** | Medium-range horizon: Day 1 (24h) to Day 10 (240h) updated at 00Z/12Z cycles |
| **Version** | 2.0 (Post-Plan Review Specification) |

---

## 2. Problem Statement & Operational Rationale

### 2.1 The Operational Challenge
Numerical Weather Prediction (NWP) systems (including IMD-GFS, NCMRWF NCUM-G at 12 km, and the 23-member NEPS ensemble) form the backbone of national meteorological forecasts. However, during rapid nonlinear transitions, deterministic and ensemble mean forecasts occasionally experience severe localized failures known as **"Forecast Busts"**.

A forecast bust is an anomalously large, flow-dependent error where the model prediction drastically diverges from observed reality, often missing the onset, intensity, or spatial trajectory of high-impact weather systems.

### 2.2 Indian Synoptic Failure Modes
In the Indian monsoon and tropical context, forecast busts are heavily structured by recurring meteorological regimes:

| Synoptic Regime | Season | Primary Failure Mode | Operational Impact |
|---|---|---|---|
| **SW Monsoon Active/Break Transitions** | Jun–Sep (JJAS) | Premature prediction of break revival; misplacement of monsoon trough axis (>2° latitude shift) | Distorts Kharif sowing advisories, reservoir inflow planning |
| **Monsoon Lows & Depressions (BoB)** | Jul–Sep | Track curvature error and under-prediction of heavy rain core on south/south-west sector | Flash floods in Odisha, Chhattisgarh, MP, Maharashtra |
| **Western Disturbances (WD)** | Oct–Apr | Rapid trough deepening and unpredicted interaction with tropical moisture | Unanticipated heavy snow/rain, landslides in HP/Uttarakhand/J&K |
| **Tropical Cyclones (BoB / Arabian Sea)** | Apr–Jun, Oct–Dec | Rapid Intensification (RI) missed; erratic recurvature (e.g., Biparjoy 2023, Amphan 2020) | Coastal evacuation lead-time compromised |
| **Indo-Gangetic Heatwaves** | Apr–Jun | Under-prediction of peak $T_{\max}$ and persistence due to dry-line / subsidence misrepresentation | Public health alerts, power grid stress |
| **Extreme Orographic / Convective Rain** | JJAS / Post-monsoon | Severe double-penalty displacement along Western Ghats and NE foothills | Urban flooding, river basin overflow |

### 2.3 Core Solution Vision
**Pratyay** provides a **calibrated, explainable forecast confidence layer** that evaluates incoming NWP forecasts against historical error behavior and synoptic precursors **12–48 hours before model execution errors manifest in reality**.

---

## 3. Product Scope & Tiering (Phased Delivery)

To guarantee scientific rigour and timely delivery for SIH 2026, the scope is tiered:

```
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 1: CORE (SIH 2026 Primary Target)                                │
│ • Season: Indian Summer Monsoon (JJAS)                                │
│ • Hazard: Rainfall Busts (Categorical & Continuous) per 36 IMD Subs   │
│ • Lead Times: Day 1 to Day 10 (24h to 240h)                           │
│ • Models: NOAA GEFSv12 Reforecast (Stationary) + IMD Daily Obs        │
│ • Engine: Model Ladder (Climatology → Spread → Analog → GBDT → U-Net) │
│ • Output: Calibrated Confidence Index + Bust Probability + XAI Cards  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 2: STRETCH 1 (Extended High-Impact Hazards)                       │
│ • Hazard: Pre-monsoon Heatwaves (April–June T_max departure)          │
│ • Hazard: Western Disturbance Winter Precipitation (Oct–March)        │
│ • Data: ERA5 Daily T_max statistics + IMD 1° Temperature Grid         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 3: STRETCH 2 (Operational Multi-Model Adapter)                   │
│ • ModelAdapter for NCMRWF NCUM-G (12 km) and NEPS (23-member ensemble)│
│ • Dedicated Oceanic Cyclone Track & Intensity Verification (RSMC/IMD) │
│ • Real-time 00Z/12Z automated ingestion pipeline                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. User Personas & Core Workflows

### 4.1 Target Users
1. **IMD Duty Forecaster (National Weather Forecasting Centre, New Delhi & RMCs)**:
   - Needs to know whether to trust Day 3–5 GFS/NCUM rainfall fields before issuing color-coded district warnings (Red/Orange/Yellow).
2. **NCMRWF Diagnostic Scientist**:
   - Needs systematic diagnostics on model weakness regimes, spread-skill deficit, and analog failure patterns.
3. **Disaster Management Officer (SDMA / NDMA)**:
   - Needs actionable probabilistic warnings with low false-alarm ratio to mobilize emergency response without "warning fatigue".

### 4.2 Key User Journeys (Workflows)
- **Workflow 1 — Daily Synoptic Risk Assessment**:
  Forecaster selects today's cycle (`00Z`) $\rightarrow$ views All-India Confidence Map for Day 1 through Day 10 $\rightarrow$ identifies red/orange subdivisions with low confidence.
- **Workflow 2 — Regional Diagnostic Deep-Dive**:
  Forecaster clicks on *East Madhya Pradesh* $\rightarrow$ inspects top 3 physical drivers (e.g., *Somali LLJ Surge mismatch*, *Large 12h run-to-run jumpiness*), historical analogs (e.g., *Monsoon Depression July 2020*), and auto-generated operational advisory.
- **Workflow 3 — Model Skill & Baseline Comparison**:
  User toggles between Raw Ensemble Spread vs. Pratyay AI Calibrated Bust Probability vs. Climatology Baseline to verify skill gain.

---

## 5. Functional Requirements (FRs)

### 5.1 Data Ingestion & Atmospheric Processing
- **FR-01 (Stationary Forecast Archive)**: Ingest NOAA GEFSv12 0.5° Reforecast (2000–2019, 00Z init, 5 control/perturbed members `c00`, `p01`–`p04`) for stationary baseline training.
- **FR-02 (Operational Evaluation Set)**: Ingest operational GEFSv12 (2021–present) for out-of-sample testing and live demonstration.
- **FR-03 (Observational Truth)**: Ingest IMD 0.25° daily gridded rainfall (1901–2024 archive; 03 UTC–03 UTC accumulation window with documented ~3h offset).
- **FR-04 (Atmospheric State Truth)**: Ingest ERA5 single-level and pressure-level reanalysis (Google ARCO-ERA5 Zarr store or monthly NetCDF) for dynamic state fields ($Z_{500}, T_{850}, q_{850}, u, v$).
- **FR-05 (Wider Input Domain)**: Atmospheric feature extraction covers the wider tropical domain ($40^\circ\text{E}–120^\circ\text{E}, 15^\circ\text{S}–50^\circ\text{N}$) to observe upstream WDs and Indian Ocean convective triggers.

### 5.2 Dynamic Feature Engineering & Synoptic Indices
- **FR-06 (True Moisture Flux Convergence)**: Compute 850 hPa MFC using specific humidity: $\text{MFC} = -\nabla \cdot (q \vec{V})$ with spherical metric scaling ($\frac{1}{R\cos\phi}\frac{\partial}{\partial\lambda}, \frac{1}{R}\frac{\partial}{\partial\phi}$).
- **FR-07 (Scaled Relative Vorticity)**: Compute 850 hPa relative vorticity $\zeta = \frac{1}{R\cos\phi}\left(\frac{\partial v}{\partial\lambda} - \frac{\partial(u\cos\phi)}{\partial\phi}\right)$ scaled to $10^{-5}\text{ s}^{-1}$.
- **FR-08 (Somali Low-Level Jet Index)**: Compute mean 850 hPa westerly wind speed over the Arabian Sea core box ($5^\circ\text{N}–15^\circ\text{N}, 60^\circ\text{E}–70^\circ\text{E}$).
- **FR-09 (Monsoon Trough Latitude Index)**: Diagnose mean latitude of minimum MSLP along the $75^\circ\text{E}–85^\circ\text{E}$ longitudinal corridor.
- **FR-10 (Run-to-Run Jumpiness / Tendency)**: Compute forecast divergence for the **same valid verification time** across consecutive NWP runs: $|F_{\text{init}=t}(\text{valid}=t+\tau) - F_{\text{init}=t-12\text{h}}(\text{valid}=t+\tau)|$.
- **FR-11 (Real Ensemble Spread)**: Extract 5-member ensemble standard deviation per variable and lead time.

### 5.3 Statistically Rigorous Bust Definition & Labeling
- **FR-12 (Subdivision-Scale Categorical Busts)**: Define rainfall busts at the 36 IMD subdivision level using IMD operational thresholds:
  - **Heavy Rain Miss (Under-prediction Bust)**: Observed $\ge 64.5\text{ mm/day}$ while Forecast $< 35.5\text{ mm/day}$.
  - **False Alarm Bust (Over-prediction Bust)**: Forecast $\ge 64.5\text{ mm/day}$ while Observed $< 15.5\text{ mm/day}$.
  - **Severe Categorical Bust**: Extreme threshold miss ($\ge 115.6\text{ mm/day}$ very heavy rain).
- **FR-13 (Percentile-Based Continuous Busts)**: Standardize continuous errors against lead-specific, month-specific training 90th percentile thresholds ($P_{90}(\text{Error})$) computed strictly on training years.
- **FR-14 (Dual Output Definition)**:
  - **Confidence Score ($0–100\%$)**: Calibrated probability that error remains within operational tolerance (naturally degrades from Day 1 to Day 10).
  - **Bust Probability ($0.0–1.0$)**: Calibrated probability of an anomalous, flow-dependent failure relative to lead-time expectations.

### 5.4 Multi-Level Machine Learning Architecture
- **FR-15 (Model Ladder Benchmark)**:
  1. *Baseline 1*: Climatological bust frequency per subdivision/lead/month.
  2. *Baseline 2*: GEFS Ensemble Spread-Skill ratio.
  3. *Baseline 3*: Historical Analog Matcher (K-nearest synoptic forecast states).
  4. *Core Model*: Sub-division LightGBM / HistGradientBoosting with blocked cross-validation.
  5. *Spatial Model*: 2D U-Net (with size $160 \times 128$ padding divisible by 32) evaluating spatial structure.
- **FR-16 (Meta-Ensemble Calibrator)**: Stacking meta-model trained on out-of-fold (OOF) predictions, calibrated with Isotonic Regression / Platt scaling on validation data.
- **FR-17 (Strict Anti-Leakage Validation)**:
  - Temporal blocked split / Leave-One-Season-Out (LOSO).
  - Embargo gap $\ge 10\text{ days}$ between consecutive seasons/splits.
  - Benchmark historical cases held out completely from training.

### 5.5 Explainability & Operational Advisories (XAI)
- **FR-18 (Context-Aware Feature Attribution)**: Pre-computed SHAP TreeExplainer attributes regional risk to physical drivers, conditioned on season, region, sign, and magnitude.
- **FR-19 (Historical Precedent Matching)**: K-NN analog search returning top 2 historical matching busts with observed synoptic consequences.
- **FR-20 (Structured Forecaster Advisories)**: Automated text generator formatting a 4-part bulletin: Synoptic Regime $\rightarrow$ Key Physical Drivers $\rightarrow$ Precedent Analogs $\rightarrow$ Operational Action Recommendation.

### 5.6 Serving, API & Dashboard
- **FR-21 (FastAPI Backend)**: Asynchronous endpoints for confidence maps, subdivision deep-dives, alerts, verification metrics, benchmarks, and GeoJSON.
- **FR-22 (Interactive GIS Dashboard)**: Leaflet.js choropleth map of 36 IMD Subdivisions + CartoDB dark base map + Day 1–10 lead time controls + diagnostic drawer + Chart.js confidence trajectory.
- **FR-23 (Model-Agnostic Adapter Interface)**: Modular `ModelAdapter` abstract base class permitting seamless extension from GEFS to NCMRWF NCUM and NEPS.

---

## 6. Non-Functional Requirements (NFRs)

| ID | Category | Metric / Constraint | Verification Method |
|---|---|---|---|
| **NFR-01** | **Inference Latency** | Full All-India Day 1–10 assessment execution $\le 3.5\text{ seconds}$ on CPU | Automated benchmark in CI/test runner |
| **NFR-02** | **Skill vs Climatology** | Brier Skill Score (BSS) $\ge 0.25$ over climatology baseline on test seasons | Blocked CV test evaluation |
| **NFR-03** | **Skill vs Ensemble Spread** | BSS $> 0$ and PR-AUC $> \text{Spread Baseline}$ | Model ladder ablation table |
| **NFR-04** | **Discrimination Skill** | Precision-Recall AUC (PR-AUC) $\ge 0.45$ and ROC-AUC $\ge 0.82$ | Test set verification curve |
| **NFR-05** | **Spatial Accuracy** | Fractions Skill Score (FSS) $\ge 0.60$ for spatial rainfall bust extent | Gridded FSS verification |
| **NFR-06** | **Calibration Quality** | Maximum Calibration Error (MCE) $\le 0.08$ on reliability diagram | Reliability curve analysis |
| **NFR-07** | **Offline Resilience** | Complete dashboard and API functional in offline standalone mode | Demo dry-run without network connection |
| **NFR-08** | **Code Quality & Testing** | 100% test pass rate across unit, data, model, and API test suites | `python tests/run_tests.py` |

---

## 7. Operational Success Criteria for SIH 2026

1. **Ablation Model Ladder Proof**: Clear demonstration that Pratyay outperforms (1) Climatology, (2) Raw Ensemble Spread, and (3) Naive Analogs.
2. **Real Case Study Validation**: Accurate early warning ($>75\%$ bust probability at Day 3–5) on held-out benchmark events (*Kerala 2018*, *Amphan 2020*, *Biparjoy 2023*, *August 2023 Monsoon Break*).
3. **Forecaster Actionability**: Duty forecasters can determine in $\le 10\text{ seconds}$ whether a subdivision's forecast is reliable and read the physical rationale.
4. **NCMRWF System Alignment**: Architecture directly supports dropping in NCUM and NEPS ensemble data streams without refactoring.

---