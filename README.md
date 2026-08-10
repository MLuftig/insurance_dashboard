# Pet Insurance Portfolio Dashboard

An interactive Streamlit dashboard giving an aggregate, at-a-glance view across 50,000 real pet insurance policies and 210,000+ real claims — population breakdown, claim severity, a Monte Carlo portfolio cost forecast, a breed-level pricing fairness audit, and survival curves, all on one screen.

**Live Dashboard:** [insurance-portfolio-dashboard.streamlit.app](https://insurance-portfolio-dashboard.streamlit.app/)

**Full methodology, model validation, and findings:** [Pet Insurance Risk & Pricing Analysis](https://github.com/MLuftig/pet-insurance-risk-and-pricing-analysis) — this repo is the deployed dashboard only; the underlying research (breed predisposition sourcing, model calibration, portfolio validation, survival analysis, and the pricing fairness audit) lives there.

**Related:** [Pet Insurance Cost Simulator](https://github.com/MLuftig/pet_insurance_cost_simulator) — the individual-pet companion tool, using the same calibrated model to simulate a single policy's plausible cost range rather than the whole portfolio's.

## What It Shows

- **Top-line KPIs** — total policies, total claims filed, ever-claimed rate, total annual premium collected
- **Population breakdown** — species split and age distribution at enrollment
- **Claim severity distribution** — the real, empirical distribution of individual claim amounts (heavily right-skewed, as real insurance claims data actually looks)
- **Monte Carlo portfolio cost forecast** — 5,000 simulated years of total portfolio claims cost, using the calibrated classifier and empirical severity bootstrap, shown as a full distribution rather than a single number. Validated to within 0.04% of the real historical total.
- **Breed-level pricing fairness audit** — the deductible- and species-adjusted pricing ratio by breed, visualized as a sorted, color-coded bar chart (green = priced closer to modeled risk, red = priced further under it)
- **Time to first claim** — Kaplan-Meier survival curves by species, showing the real pace at which policies file their first claim, not just whether they do

## Design Note
Every figure on this dashboard is pulled directly from the same calibrated model and validated computations as the linked analysis repo — nothing here is recalculated with different assumptions or a simplified version of the methodology. The dashboard's purpose is presentation of already-validated findings at a glance, not a separate analysis.

## Running Locally
```bash
git clone https://github.com/MLuftig/insurance_dashboard.git
cd insurance_dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
`Python`, `Streamlit`, `Plotly`, `Scikit-Learn`, `Joblib`, `lifelines`, `pandas`, `NumPy`

## Repository Structure
```text
├── app.py                                     # Streamlit dashboard application
├── PetData.csv
├── ClaimData.csv
├── breed_predisposition_lookup.py             # Real, cited breed-to-condition mapping
├── pet_insurance_claims_classifier.pkl        # Pre-trained, calibrated Random Forest classifier
├── real_severity_pool.npy                     # Empirical claim-severity data for Monte Carlo bootstrap sampling
├── requirements.txt
└── README.md
```
