# PART IV: Design Specification & UI/UX Reference

## 1. Overview & Status

> [!NOTE]
> **Status: Awaiting User Reference Design**  
> As requested, this document serves as the UI/UX design reference for **Project Pratyay**. The baseline operational layout is currently implemented and running in [`src/api/dashboard.html`](file:///C:/Users/Vivek/.gemini/antigravity/scratch/forecast-bust-detector/src/api/dashboard.html).  
> When you have your reference design ready (mockups, Figma links, wireframes, or specific UI rules), share them and they will be incorporated directly into this document and applied to the dashboard codebase.

---

## 2. Current Baseline Design Layout (Implemented)

The operational dashboard currently runs at `http://localhost:8000/dashboard` with the following structural layout:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  🌦️ Satark-Mausam / Pratyay (प्रत्यय)         [Scenario Selector] [Date Input] [API Docs]       │
│  Ministry of Earth Sciences (MoES) / NCMRWF · 00Z Cycle · OPERATIONAL                            │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Forecast Lead Time: [Day 1] [Day 2] [Day 3] [Day 4] [Day 5] [Day 6] [Day 7] [Day 8] [Day 9] [Day 10] │
│  Homogeneous Overview: [NWI: 45.8%] [CNI: 54.0%] [SPI: 54.0%] [ENE: 54.0%]                        │
├────────────────────────────────────────────────────┬─────────────────────────────────────────────┤
│                                                    │  📋 FORECASTER DIAGNOSTIC DESK              │
│   INDIA 36-SUBDIVISION CONFIDENCE MAP             │  ┌────────────────────────────────────────┐ │
│   (Interactive Leaflet GIS + CartoDB Dark Base)    │  │ SUB_35 · Kerala & Mahe                 │ │
│                                                    │  │ Confidence: 82.4%  HIGH CONFIDENCE     │ │
│   🟢 High Confidence (>75%)                        │  └────────────────────────────────────────┘ │
│   🟡 Moderate Confidence (50–75%)                  │                                             │
│   🟠 Low Confidence (25–50%)                       │  📈 Day 1 → Day 10 Confidence Trajectory    │
│   🔴 Severe Bust Warning (<25%)                    │  ┌────────────────────────────────────────┐ │
│                                                    │  │ [Chart.js Uncertainty Growth Curve]    │ │
│   • Hover: Instant tooltip (Bust Prob & Level)     │  └────────────────────────────────────────┘ │
│   • Click: Instant drawer population               │                                             │
│                                                    │  🌪️ Identified Synoptic Regimes           │
│                                                    │  • Somali LLJ Surge & Offshore Trough      │
│                                                    │                                             │
│                                                    │  📊 Key Physical Drivers (SHAP Values)     │
│                                                    │  • High 850 hPa Moisture Flux (+4.2)       │
│                                                    │  • Somali Jet Speed > 18 m/s (+3.8)        │
│                                                    │                                             │
│                                                    │  🏛️ Precedent Historical Bust Analogs       │
│                                                    │  • Kerala Floods Aug 2018 (98% Match)      │
│                                                    │                                             │
│                                                    │  📢 Duty Forecaster Operational Advisory   │
│                                                    │  • 4-Section Standardized Bulletin Text    │
├────────────────────────────────────────────────────┴─────────────────────────────────────────────┤
│  Verification Scorecard: BSS: 0.284 | ROC-AUC: 0.862 | Hit Rate: 79.5% | FAR: 23.1%              │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Color Palette & Visual Hierarchy

### Operational Alert Color Scale (WMO / IMD Standard)
| Alert Category | Color Name | Hex Code | Background Tint | Condition |
|---|---|---|---|---|
| **HIGH CONFIDENCE** | Emerald Green | `#10b981` | `bg-emerald-950` | $P_{\text{bust}} < 0.25$ / $\text{Confidence} > 75\%$ |
| **MODERATE WATCH** | Amber Yellow | `#eab308` | `bg-yellow-950` | $0.25 \le P_{\text{bust}} < 0.50$ / $\text{Confidence } 50\text{–}75\%$ |
| **LOW CONFIDENCE ALERT** | Solar Orange | `#f97316` | `bg-amber-950` | $0.50 \le P_{\text{bust}} < 0.75$ / $\text{Confidence } 25\text{–}50\%$ |
| **CRITICAL BUST WARNING** | Rose Red | `#f43f5e` | `bg-rose-950` | $P_{\text{bust}} \ge 0.75$ / $\text{Confidence} < 25\%$ |

### Theme Colors
- **App Background:** Slate-950 (`#020617`)
- **Card Surfaces:** Slate-900 (`#0f172a`)
- **Borders & Dividers:** Slate-800 (`#1e293b`)
- **Primary Accent / Brand:** Blue-600 (`#2563eb`) & Sky-400 (`#38bdf8`)
- **Typography:** Inter / Roboto font family; high-contrast white & slate-300 text

---

## 4. Key UI Components & Interactions

1. **All-India Subdivision Map (`ConfidenceMap`)**:
   - Leaflet.js with CartoDB dark tiles (`cartocdn.com/dark_nolabels`).
   - Renders 36 IMD subdivision GeoJSON polygons.
   - Dynamic fill color driven by selected lead time (Day 1–10).
   - Instant click listener binding to `/api/v1/subdivision/{id}`.
2. **Lead Time Button Bar (`LeadTimeScrubber`)**:
   - 10 distinct pills for Day 1 (24h) through Day 10 (240h).
   - Active state highlighted in solid blue with bold weight.
   - Triggers asynchronous map color updates without full page reloads.
3. **Scenario Switcher (`ScenarioSelector`)**:
   - Quick-access dropdown to switch between *"Operational Real-Time"* and benchmark validation cases (*Kerala Floods 2018*, *Cyclone Amphan 2020*, *Cyclone Biparjoy 2023*, *August 2023 Monsoon Break*).
4. **Diagnostic Drawer (`ForecasterDrawer`)**:
   - Displays selected subdivision details, status badge, trajectory line graph, SHAP driver bars, analog cards, and formatted text bulletin.
5. **Confidence Trajectory Line Chart (`TrajectoryCurve`)**:
   - Chart.js spline curve showing confidence score from Day 1 to Day 10 with colored data points matching the warning level.

---

## 5. Reference Design Placeholder (For User Customization)

```markdown
<!-- TO BE POPULATED WHEN USER PROVIDES DESIGN REFERENCE -->
- Custom Layout Wireframe: [Insert details / Figma URL]
- Custom Component Specs: [Insert details]
- Custom Branding / Logo Assets: [Insert details]
- Mobile / Responsive Constraints: [Insert details]
- Dedicated Charts / Visualizations: [Insert details]
```

> **Note for the User:** Whenever you're ready to share your reference design (whether it's a sketch, screenshot description, UI wireframe, or specific component guidelines), let me know and I will update `Design.md` and adjust `src/api/dashboard.html` to match it!