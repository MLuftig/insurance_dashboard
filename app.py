import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from lifelines import KaplanMeierFitter
from breed_predisposition_lookup import build_breed_predisposition_table

st.set_page_config(page_title="Pet Insurance Portfolio Dashboard", page_icon="📊", layout="wide")

AGE_MAP = {'0-7 weeks old': 0.07, '8 weeks to 12 months old': 0.5,
    '1 year old': 1, '2 years old': 2, '3 years old': 3, '4 years old': 4,
    '5 years old': 5, '6 years old': 6, '7 years old': 7, '8 years old': 8,
    '9 years old': 9, '10 years old': 10, '11 years old': 11, '12 years old': 12, '13 years old': 13}
FLAG_COLS = ['risk_cancer', 'risk_orthopedic', 'risk_patellar_luxation', 'risk_cardiac',
             'risk_brachycephalic', 'risk_neurological', 'risk_drug_sensitivity',
             'risk_eye_disease', 'risk_metabolic']


@st.cache_resource
def load_model_artifacts():
    clf = joblib.load("pet_insurance_claims_classifier.pkl")
    severity_pool = np.load("real_severity_pool.npy")
    return clf, severity_pool


@st.cache_data
def load_and_prepare_data():
    clf, severity_pool = load_model_artifacts()

    pets = pd.read_csv("PetData.csv")
    claims = pd.read_csv("ClaimData.csv")
    pets["EnrollDate"] = pd.to_datetime(pets["EnrollDate"])
    claims["ClaimDate"] = pd.to_datetime(claims["ClaimDate"])
    pets = build_breed_predisposition_table(pets)

    pets["age_years"] = pets["PetAge"].map(AGE_MAP)
    for col in FLAG_COLS:
        pets[col] = pets[col].astype("float")

    pets_encoded = pd.get_dummies(pets, columns=["Species", "EnrollPath"], drop_first=True)
    pets_encoded["research_coverage"] = pets_encoded["research_coverage"].astype(int)
    feature_cols = ["age_years"] + FLAG_COLS + ["research_coverage"] + \
                    [c for c in pets_encoded.columns if c.startswith("Species_") or c.startswith("EnrollPath_")]

    claim_probs = clf.predict_proba(pets_encoded[feature_cols])[:, 1]
    pets["claim_prob"] = claim_probs
    pets["expected_cost"] = claim_probs * severity_pool.mean()
    pets["annual_premium"] = pets["Premium"] * 12
    pets["is_dog"] = (pets["Species"] == "Dog").astype(int)

    lr = LinearRegression()
    lr.fit(pets[["Deductible", "is_dog"]], pets["annual_premium"])
    pets["premium_residual"] = pets["annual_premium"] - lr.predict(pets[["Deductible", "is_dog"]])

    # First claim date, for the survival panel
    first_claim = claims.groupby("PetId")["ClaimDate"].min().reset_index()
    first_claim.columns = ["PetId", "first_claim_date"]
    pets = pets.merge(first_claim, on="PetId", how="left")
    observation_end = claims["ClaimDate"].max()
    pets["event_observed"] = pets["first_claim_date"].notna().astype(int)
    pets["duration_days"] = np.where(
        pets["event_observed"] == 1,
        (pets["first_claim_date"] - pets["EnrollDate"]).dt.days,
        (observation_end - pets["EnrollDate"]).dt.days,
    )
    pets["duration_days"] = pets["duration_days"].clip(lower=0)

    return pets, claims, severity_pool


@st.cache_data
def compute_breed_audit(_pets):
    audit = _pets[_pets["research_coverage"] == True].groupby("Breed").agg(
        n_pets=("PetId", "count"),
        avg_premium_residual=("premium_residual", "mean"),
        avg_expected_cost=("expected_cost", "mean"),
    ).reset_index()
    audit["pricing_ratio"] = audit["avg_premium_residual"] / audit["avg_expected_cost"]
    return audit.sort_values("pricing_ratio", ascending=False)


@st.cache_data
def run_monte_carlo(_pets, _severity_pool, n_sims=3000, seed=42):
    rng = np.random.default_rng(seed)
    probs = _pets["claim_prob"].values
    results = []
    for _ in range(n_sims):
        claims_occur = rng.random(len(probs)) < probs
        n_claims = claims_occur.sum()
        total = _severity_pool[rng.integers(0, len(_severity_pool), n_claims)].sum() if n_claims > 0 else 0.0
        results.append(total)
    return np.array(results)


pets, claims, severity_pool = load_and_prepare_data()
breed_audit = compute_breed_audit(pets)

st.title("Pet Insurance Portfolio Dashboard")
st.caption(
    "Aggregate view across 50,000 real pet insurance policies and 210,000+ real claims. "
    "All figures reuse the calibrated model and Monte Carlo method validated in the "
    "[full analysis](https://github.com/MLuftig/pet-insurance-risk-and-pricing-analysis)."
)

# ============================================================
# KPI row
# ============================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total policies", f"{len(pets):,}")
k2.metric("Total claims filed", f"{len(claims):,}")
k3.metric("Ever-claimed rate", f"{claims['PetId'].nunique() / len(pets):.1%}")
k4.metric("Total annual premium", f"${pets['annual_premium'].sum() / 1e6:.1f}M")

st.divider()

# ============================================================
# Population breakdown
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Population by species")
    species_counts = pets["Species"].value_counts().reset_index()
    species_counts.columns = ["Species", "Count"]
    fig = px.pie(species_counts, names="Species", values="Count", hole=0.5,
                 color_discrete_sequence=["#4C72B0", "#DD8452"])
    fig.update_traces(textinfo="label+percent")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Age distribution at enrollment")
    age_data = pets[pets["age_years"].notna()]
    fig = px.histogram(age_data, x="age_years", nbins=20, color_discrete_sequence=["#55A868"])
    fig.update_layout(xaxis_title="Age (years)", yaxis_title="Number of pets", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# Claims severity
# ============================================================
st.subheader("Claim severity distribution")
st.caption(
    f"Real, individual claim amounts (n={len(severity_pool):,}). Heavily right-skewed — "
    f"median ${np.median(severity_pool):,.0f}, mean ${severity_pool.mean():,.0f}, "
    f"max ${severity_pool.max():,.0f}."
)
fig = px.histogram(x=severity_pool[severity_pool < 5000], nbins=60, color_discrete_sequence=["#C44E52"])
fig.update_layout(xaxis_title="Claim amount ($, capped at $5,000 for readability)", yaxis_title="Number of claims", bargap=0.05)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# Monte Carlo portfolio forecast
# ============================================================
st.subheader("Monte Carlo portfolio cost forecast")
st.caption(
    "5,000 simulated years of total portfolio claims cost, using the calibrated classifier "
    "and empirical severity bootstrap. Validated to within 0.04% of the real historical total."
)
with st.spinner("Running Monte Carlo simulation..."):
    mc_results = run_monte_carlo(pets, severity_pool)

fig = go.Figure()
fig.add_trace(go.Histogram(x=mc_results, nbinsx=60, marker_color="#8172B3"))
p5, p50, p95, p99 = np.percentile(mc_results, [5, 50, 95, 99])
for val, label, color in [(p50, "Median", "#333"), (p95, "95th pct", "#DD8452"), (p99, "99th pct ('bad year')", "#C44E52")]:
    fig.add_vline(x=val, line_dash="dash", line_color=color, annotation_text=label)
fig.update_layout(xaxis_title="Total annual portfolio cost ($)", yaxis_title="Simulated years", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("5th percentile", f"${p5/1e6:.2f}M")
m2.metric("Median", f"${p50/1e6:.2f}M")
m3.metric("95th percentile", f"${p95/1e6:.2f}M")
m4.metric("99th percentile", f"${p99/1e6:.2f}M")

st.divider()

# ============================================================
# Breed pricing fairness audit
# ============================================================
st.subheader("Breed-level pricing fairness audit")
st.caption(
    "Deductible- and species-adjusted premium residual, divided by model-predicted expected cost. "
    "Ratio near 1 means pricing tracks predicted risk; below 0 means priced under the deductible/species baseline."
)
fig = px.bar(
    breed_audit, x="pricing_ratio", y="Breed", orientation="h",
    color="pricing_ratio", color_continuous_scale=["#C44E52", "#DDD", "#55A868"],
    color_continuous_midpoint=0,
)
fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Pricing ratio", coloraxis_showscale=False, height=600)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# Survival curves
# ============================================================
st.subheader("Time to first claim, by species")
st.caption("Kaplan-Meier survival curve — probability a policy has not yet filed a claim, by days since enrollment.")

kmf = KaplanMeierFitter()
fig = go.Figure()
for species, color in [("Dog", "#4C72B0"), ("Cat", "#DD8452")]:
    mask = pets["Species"] == species
    kmf.fit(pets.loc[mask, "duration_days"], event_observed=pets.loc[mask, "event_observed"], label=species)
    sf = kmf.survival_function_.reset_index()
    fig.add_trace(go.Scatter(x=sf.iloc[:, 0], y=sf.iloc[:, 1], mode="lines", name=species, line=dict(color=color)))
fig.update_layout(xaxis_title="Days since enrollment", yaxis_title="Probability of no claim yet")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Full methodology, model correction notes, and honest limitations: "
    "[Pet Insurance Risk & Pricing Analysis](https://github.com/MLuftig/pet-insurance-risk-and-pricing-analysis). "
    "Individual-pet cost simulator: [Pet Insurance Cost Simulator](https://pet-insurance-cost-simulator.streamlit.app/)."
)
