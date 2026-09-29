HACKATHON WORKFLOW

┌────────────────────────────────────────────────────────────┐
│ WORKSTREAM 1: DATA & FEATURE ENGINEERING                   │
└────────────────────────────────────────────────────────────┘

ENTSO-E datasets
       │
       ▼

Data quality & coverage assessment
       │
       ▼

Target definition

  Activated Balancing Energy Total
  (per bidding zone, per settlement period)

       │
       ▼

Day-ahead forecasting feature set

  Demand
  - Day-ahead load forecast
  - Load ramp day-ahead

  Renewables
  - Onshore wind day-ahead
  - Offshore wind day-ahead
  - Solar day-ahead
  - Renewable share day-ahead
  - Renewables ramp day-ahead

  System conditions
  - Residual load day-ahead
  - Unavailable total
  - Unavailable thermal
  - Unavailable nuclear
  - Unavailable hydro

  Market
  - Day-ahead price

  Cross-border
  - Net position day-ahead
  - Offered transfer capacities

       │
       ▼

Derived feature candidates

  - Residual load ramps
  - Renewable penetration bins
  - Outage severity metrics
  - Import dependency proxies
  - Price × residual load interactions
  - Renewable share × outage interactions
  - Transfer capacity × net position interactions
  - Residual load × outage interactions
  - Ramp × outage interactions

─────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────┐
│ WORKSTREAM 2: BENCHMARKS                                   │
└────────────────────────────────────────────────────────────┘

Simple deterministic forecasts

  - Zero balancing forecast
  - Previous day same period
  - Historical average for SP
  - Historical average by hour/day type

These establish whether any ML model is
adding genuine predictive value.

─────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────┐
│ WORKSTREAM 3: QUANTILE MODELLING                           │
└────────────────────────────────────────────────────────────┘

Objective

  Forecast the day-ahead balancing volume
  distribution for each settlement period

                X(t)
                 │
                 ▼

      q50(Balancing Volume)

  Candidate drivers

    Demand
    - Day-ahead load forecast
    - Load ramp day-ahead

    System state
    - Residual load day-ahead
    - Renewable share day-ahead

    Availability
    - Unavailable thermal
    - Unavailable nuclear

    Market
    - Day-ahead price

    Calendar
    - Hour
    - Day type
    - Season

  Hypothesis

    Expected balancing requirements are
    primarily driven by demand, residual load
    and generation availability.

                 │
                 ▼

      q95(Balancing Volume)

  Candidate drivers

    Core system drivers
    - Day-ahead load forecast
    - Residual load day-ahead
    - Unavailable thermal
    - Unavailable nuclear

    Variability drivers
    - Renewables ramp day-ahead
    - Load ramp day-ahead
    - Renewable share day-ahead

    Cross-border drivers
    - Net position day-ahead
    - Offered transfer capacities

    Interaction terms
    - Renewable share × outages
    - Residual load × outages
    - Renewable share × transfer capacity
    - Ramp × outages

  Rationale

    Large renewable ramps increase the need
    for flexibility and reserve actions.

    High renewable penetration may increase
    forecast uncertainty and reduce the share
    of dispatchable generation.

    Outages reduce available flexibility and
    can amplify the impact of unexpected
    demand or renewable deviations.

    Limited transfer capacity may restrict
    access to balancing resources from
    neighbouring markets.

  Hypothesis

    Extreme balancing requirements arise
    when normal system drivers (demand and
    residual load) coincide with reduced
    flexibility, high variability or
    constrained access to balancing resources.

                 │
                 ▼

      Quantile forecasts

        q50(t)

        q95(t)

                 │
                 ▼

      Stress metrics

        q95 - q50

            or

        q95 / q50

─────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────┐
│ WORKSTREAM 4: UNCERTAINTY ASSESSMENT                       │
└────────────────────────────────────────────────────────────┘

Multiple learning methods

  - XGBoost Quantile Regression
  - Quantile Random Forest
  - Linear/GAM Quantile Model

                 │
                 ▼

Bootstrap resampling

                 │
                 ▼

Assess source of uncertainty

  Large q95-q50 spread
  + Stable across models
  + Stable across bootstraps

  ⇒ Aleatoric uncertainty
     (genuine system risk)

  Large q95-q50 spread
  + Unstable across models
  + Unstable across bootstraps

  ⇒ Epistemic uncertainty
     (limited data / model uncertainty)

─────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────┐
│ WORKSTREAM 5: VALIDATION                                   │
└────────────────────────────────────────────────────────────┘

Deterministic validation

  - MAE
  - RMSE
  - Skill score vs benchmarks

Probabilistic validation

  - Quantile coverage
  - Calibration / reliability analysis

      q95 should be exceeded
      approximately 5% of the time

  - Pinball loss
  - Coverage by season
  - Coverage by hour of day

─────────────────────────────────────────────────────────────
┌───────────────────────────────────────────────────────────────────┐
│ WORKSTREAM 6: EX-POST ANALYSIS | EXPLAINABILITY & BUSINESS VALUE  │
└───────────────────────────────────────────────────────────────────┘

Model forecasts

    q50(t)
    q95(t)

          │
          ▼

Forecast stress metric

    q95 - q50

       or

    q95 / q50

    Large value
      = elevated uncertainty
      = elevated tail risk

          │
          ▼

Explainability analysis

    SHAP(q50 model)

        • What drives normal balancing needs?

    SHAP(q95 model)

        • What drives extreme balancing needs?

    SHAP(stress metric)

        • What drives uncertainty?
        • What widens the upper tail?

          │
          ▼

Driver categorisation

    Renewable conditions
        - Forecast renewable output
        - Renewable ramps
        - Renewable share

    Demand conditions
        - Forecast load
        - Demand ramps

    System conditions
        - Thermal outages
        - Nuclear outages
        - Transfer constraints

    Market conditions
        - Day-ahead prices
        - Spread indicators
        - Scarcity signals

    Cross-border conditions
        - Import dependency
        - Neighbouring system stress

          │
          ▼

Operational interpretation

Question:

    "Why is tomorrow risky?"

Example output:

    q95 elevated because:

        + High renewable penetration
        + Large forecast wind ramp
        + Reduced thermal availability
        + High import dependency

          │
          ▼

Translate to TSO signals

    Tail-risk monitor

    Reserve adequacy monitor

    Market stress monitor

    System flexibility monitor

          │
          ▼

Operational actions / applications

    Day-ahead risk briefing

    Enhanced situational awareness

    Challenge reserve assumptions

    Escalate high-risk periods
    for operational review

    Target further investigation
    into risk drivers

          │
          ▼

Post-event learning

Did the risk materialise?

    YES
       -> which drivers mattered?
       -> were warnings correct?

    NO
       -> false alarm?
       -> feature review?
       -> recalibration?

          │
          ▼

Business value assessment

    Earlier identification of
    system stress periods

    Improved reserve planning

    Better understanding of
    balancing risk drivers

    Increased confidence and
    trust in ML forecasts

    Explainable evidence for
    operational decision support

─────────────────────────────────────────────────────────────

FINAL OUTPUT

Day-ahead balancing risk forecast
for every settlement period

     q50(t)

        +

     q95(t)

        ▼

Stress indicator

     q95/q50

       or

     q95-q50

        ▼

Explainable risk drivers

        ▼

Actionable operational insight
for TSO decision support