# PART III: Phases — Build Roadmap & Execution Milestones

## 1. Roadmap Overview

```
Phase 0 ──► Scientific Setup, Architecture & Data Contracts (COMPLETE)
Phase 1 ──► Data Acquisition, ModelAdapter & Error Climatology (COMPLETE)
Phase 2 ──► Feature Engineering & Baseline Model Ladder (COMPLETE)
Phase 3 ──► Core ML Engine, OOF Stacking & Calibration (COMPLETE)
Phase 4 ──► Context-Aware Explainability & Advisory Synthesis (COMPLETE)
Phase 5 ──► Operational FastAPI Backend & Interactive GIS Dashboard (COMPLETE)
Phase 6 ──► SIH Evaluation, Benchmark Case Studies & Demo Polish (ACTIVE)
```

---

## 2. Detailed Phase Breakdown

### Phase 0: Scientific Setup, Architecture & Data Contracts
- **Objective:** Establish the scientifically sound foundation for the project incorporating all plan review fixes.
- **Key Deliverables:**
  - `PRD.md`: Formal product specifications, tiered scope, operational user journeys.
  - `architecture.md`: System design, spherical derivative formulations, progressive model ladder, data flow.
  - `rules.md`: Scientific rules (anti-leakage, temporal splitting, calibration, IMD thresholds).
  - `src/config.py`: Complete domain parameters, 36 IMD subdivisions, 4 homogeneous regions, benchmark case definitions.
  - `requirements.txt` & `pyproject.toml`: Environment definitions with pinned scientific dependencies.
- **Verification:** All configuration constants and spatial bounds verified against official IMD documentation.
- **Status:** ✅ **COMPLETE**

---

### Phase 1: Data Acquisition, ModelAdapter & Error Climatology
- **Objective:** Implement data ingestion pipeline supporting stationary training data and operational live streams.
- **Key Deliverables:**
  - `src/data/fetcher.py`: Ingestion logic for NOAA GEFSv12 Reforecast (`s3://noaa-gefs-retrospective`), operational GEFSv12, and offline simulation fallback.
  - `src/data/regions.py`: Spatial masking and GeoJSON generator for 36 IMD Subdivisions and 4 Homogeneous Regions.
  - `src/data/preprocessor.py`: Implementation of spherical metric relative vorticity $\zeta_{850}$, true moisture flux convergence $-\nabla\cdot(q\vec{V})$, vertical wind shear, and training-only $P_{90}$ standardized error baselines.
- **Verification:** Tests 01, 02, 03 in `tests/run_tests.py` passing.
- **Status:** ✅ **COMPLETE**

---

### Phase 2: Feature Engineering & Baseline Model Ladder
- **Objective:** Extract synoptic indices and build the lower rungs of the model ladder.
- **Key Deliverables:**
  - `src/data/feature_engineer.py`: Extraction of Somali LLJ index ($5^\circ\text{–}15^\circ\text{N}, 60^\circ\text{–}70^\circ\text{E}$), Monsoon Trough latitude, WD trough depth, MJO RMM indices, log-transformed precipitation moments, and run-to-run forecast jumpiness $\Delta F_{12\text{h}}$.
  - **Baseline 1:** Climatological bust frequency calculator per subdivision, lead time, and month.
  - **Baseline 2:** GEFS 5-member ensemble spread-skill ratio evaluator ($\sigma_{\text{ens}} / \text{RMSE}_{\text{clim}}$).
  - **Baseline 3:** K-NN Historical Analog matching engine.
- **Verification:** Test 04 in `tests/run_tests.py` passing (36 feature rows with climate indices).
- **Status:** ✅ **COMPLETE**

---

### Phase 3: Core ML Engine, OOF Stacking & Probability Calibration
- **Objective:** Implement dual-branch machine learning architecture with strict probability calibration.
- **Key Deliverables:**
  - `src/models/losses.py`: Focal Loss ($\alpha=0.75, \gamma=2.0$) and positive-weighted BCE with logits.
  - `src/models/spatial_model.py`: 2D U-Net (padded to $160 \times 128$) with spatial attention, Grad-CAM hooks, and CPU/NumPy fallback.
  - `src/models/tabular_model.py`: Subdivision-level LightGBM / HistGradientBoosting with blocked cross-validation and evaluation metrics (BSS, PR-AUC, ROC-AUC, CSI).
  - `src/models/ensemble.py`: Meta-ensemble stacking trained on Out-of-Fold (OOF) predictions, calibrated with Isotonic Regression.
- **Verification:** Test 05 in `tests/run_tests.py` passing (BSS, ROC-AUC $\ge 0.82$, FAR $< 30\%$).
- **Status:** ✅ **COMPLETE**

---

### Phase 4: Context-Aware Explainability & Advisory Synthesis
- **Objective:** Build multi-level XAI translating mathematical activations into operational meteorological rationale.
- **Key Deliverables:**
  - `src/explainability/explainer.py`: Pre-computed SHAP TreeExplainer conditioned on season, region, sign, and magnitude; Indian synoptic regime classifier.
  - `src/explainability/analog_finder.py`: Historical analog search referencing landmark failure cases.
  - `src/explainability/templates.py`: Automated generator of 4-section structured operational advisories.
- **Verification:** Test 06 in `tests/run_tests.py` passing (Kerala flood analog matched with $\ge 95\%$ similarity).
- **Status:** ✅ **COMPLETE**

---

### Phase 5: Operational FastAPI Backend & Interactive GIS Dashboard
- **Objective:** Connect the end-to-end pipeline into a high-performance REST API and single-page Leaflet dashboard.
- **Key Deliverables:**
  - `src/pipeline/inference.py`: End-to-end Day 1 to Day 10 pipeline orchestrator.
  - `src/api/schemas.py`: Pydantic v2 schemas for all API payloads.
  - `src/api/main.py`: FastAPI server with 8 operational endpoints.
  - `src/api/dashboard.html`: Single-page interactive GIS dashboard (Leaflet.js + Tailwind CSS + Chart.js) with Day 1–10 slider, scenario selector, diagnostic drawer, and live trajectory chart.
- **Verification:** Tests 01–08 in `tests/test_api_endpoints.py` passing with 100% OK.
- **Status:** ✅ **COMPLETE**

---

### Phase 6: SIH Evaluation, Benchmark Case Studies & Demo Polish
- **Objective:** Validate system on real historical benchmark events and prepare final SIH presentation deliverables.
- **Active Tasks:**
  - [ ] Validate on 4 landmark case studies:
    1. *Kerala Floods (August 2018)*
    2. *Super Cyclone Amphan (May 2020)*
    3. *Very Severe Cyclone Biparjoy (June 2023)*
    4. *Monsoon Extended Break Spell (August 2023)*
  - [ ] Incorporate custom UI design reference from `Design.md` once provided by user.
  - [ ] Generate comprehensive ablation summary table (Climatology vs. Spread vs. Analogs vs. LightGBM vs. Stacking).
  - [ ] Package frozen offline demonstration cache for reliable evaluation.
  - [ ] Draft SIH presentation pitch deck outline.
- **Status:** 🟡 **IN PROGRESS (ACTIVE PHASE)**

---

## 3. Phase Status Summary Table

| Phase | Title | Target Timeline | Status | Automated Test Pass |
|---|---|---|---|---|
| **Phase 0** | Scientific Setup & Data Contracts | Day 0 | ✅ Complete | Verified |
| **Phase 1** | Data Ingestion & Atmospheric Dynamics | Week 1 | ✅ Complete | 3/3 Tests |
| **Phase 2** | Feature Engineering & Baseline Ladder | Week 2 | ✅ Complete | 1/1 Test |
| **Phase 3** | Core ML Engine & Calibration | Week 3 | ✅ Complete | 1/1 Test |
| **Phase 4** | Context-Aware Explainability | Week 4 | ✅ Complete | 1/1 Test |
| **Phase 5** | REST API & Interactive GIS Dashboard | Week 5 | ✅ Complete | 8/8 Tests |
| **Phase 6** | SIH Evaluation & Benchmark Validation | Week 6 | 🟡 Active | Pending Design.md |

---