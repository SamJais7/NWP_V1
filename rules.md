# Scientific & Engineering Rules for Project Pratyay (Satark-Mausam)

## 1. Temporal Anti-Leakage & Validation Rules
1. **Stationary Training vs. Operational Testing**:
   - Training partition is strictly restricted to NOAA GEFSv12 0.5° Reforecast archive (2000–2019 JJAS, 00Z cycle).
   - Operational evaluation partition uses independent 2021–2024 operational cycles.
2. **Blocked Leave-One-Season-Out (LOSO) Cross-Validation**:
   - Cross-validation must be blocked by monsoon season (JJAS). Never perform random row shuffle on temporal atmospheric time-series.
3. **Embargo Gap**:
   - A temporal buffer of $\ge 10\text{ days}$ must separate training folds from evaluation folds to prevent autocorrelation leakage from long-lived synoptic systems.
4. **Benchmark Case Study Isolation**:
   - The 4 landmark benchmark events (*Kerala Floods Aug 2018*, *Super Cyclone Amphan May 2020*, *Cyclone Biparjoy Jun 2023*, *August 2023 Monsoon Break*) are completely quarantined and held out from model fitting.

---

## 2. Atmospheric Dynamics & Scientific Formulations
1. **Spherical Metric Coordinates**:
   - All spatial derivatives on the latitude-longitude grid must account for Earth's spherical curvature:
     $$\frac{\partial}{\partial x} \to \frac{1}{R\cos\phi}\frac{\partial}{\partial\lambda}, \quad \frac{\partial}{\partial y} \to \frac{1}{R}\frac{\partial}{\partial\phi}$$
     where $R = 6.371 \times 10^6\text{ m}$.
2. **True Moisture Flux Convergence (MFC)**:
   - Must be computed using specific humidity $q_{850}$ (kg/kg) and horizontal winds ($u_{850}, v_{850}$), NOT relative humidity:
     $$\text{MFC} = -\nabla \cdot (q_{850} \vec{V}_{850})$$
3. **Scaled Relative Vorticity ($\zeta_{850}$)**:
   - Evaluated as $\frac{1}{R\cos\phi}\left(\frac{\partial v_{850}}{\partial\lambda} - \frac{\partial(u_{850}\cos\phi)}{\partial\phi}\right) \times 10^5\text{ s}^{-1}$.
4. **Vertical Wind Shear (200–850 hPa)**:
   - Evaluated as $\sqrt{(u_{200} - u_{850})^2 + (v_{200} - v_{850})^2}$ in m/s.
5. **Run-to-Run Forecast Jumpiness ($\Delta F_{12\text{h}}$)**:
   - Evaluated for identical valid verification time across consecutive runs:
     $$\Delta F_{12\text{h}}(s, t+\tau) = |F_{\text{init}=t}(s, t+\tau) - F_{\text{init}=t-12\text{h}}(s, t+\tau)|$$

---

## 3. Operational Bust Criteria (IMD Scale)
1. **Categorical Rainfall Busts**:
   - **Heavy Rain Miss (Under-forecast)**: Observed $\ge 64.5\text{ mm/day}$ while NWP Forecast $< 35.5\text{ mm/day}$.
   - **False Alarm (Over-forecast)**: NWP Forecast $\ge 64.5\text{ mm/day}$ while Observed $< 15.5\text{ mm/day}$.
   - **Severe Categorical Miss**: Observed $\ge 115.6\text{ mm/day}$ (Very Heavy Rain) with $<50\%$ forecast capture.
2. **Percentile-Based Continuous Busts**:
   - $Y = 1$ if $|F - O| \ge P_{90}(|F_{\text{train}} - O_{\text{train}}|)_{s, \text{month}, \tau}$. Percentiles are derived strictly from training years.
3. **Dual Output Protocol**:
   - **Confidence Score (0–100%)**: Monotonically degrades with lead time (Day 1 $\to$ Day 10).
   - **Bust Probability (0.0–1.0)**: Calibrated flow-dependent risk relative to expected lead-time baseline.

---

## 4. Probability Calibration & Stacking
1. **Out-of-Fold (OOF) Training**:
   - Meta-ensemble stacking and post-hoc probability calibrators (Isotonic Regression) must be fit solely on out-of-fold validation predictions to avoid over-confident calibration.
