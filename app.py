import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Pro Football AI Engine", layout="centered")
st.title("⚽ Pro Football AI Engine")
st.subheader("📊 Match Predictor (Attack/Defence + Dixon-Coles)")

MAX_G = 8


# ---------- Model ----------
def per_match(xg, goals, n, w_xg):
    """Blend xG and actual goals into a per-match rate (xG is more stable)."""
    return (w_xg * xg + (1 - w_xg) * goals) / n


def shrink(rate, league_avg, n, k):
    """Pull small-sample rates toward the league average."""
    return (n * rate + k * league_avg) / (n + k)


def dc_tau(i, j, lh, la, rho):
    if i == 0 and j == 0:
        return 1 - lh * la * rho
    if i == 0 and j == 1:
        return 1 + lh * rho
    if i == 1 and j == 0:
        return 1 + la * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def score_matrix(lh, la, rho):
    ph = poisson.pmf(np.arange(MAX_G + 1), lh)
    pa = poisson.pmf(np.arange(MAX_G + 1), la)
    m = np.outer(ph, pa)
    for i in range(2):
        for j in range(2):
            m[i, j] *= dc_tau(i, j, lh, la, rho)
    return m / m.sum()


# ---------- Sidebar ----------
sb = st.sidebar
sb.header("Match Data Input")
home_team = sb.text_input("Home Team", "Barcelona")
away_team = sb.text_input("Away Team", "Real Madrid")

sb.markdown("---")
sb.subheader("🏆 League settings")
lg_home = sb.number_input("League avg home goals/match", 0.5, 3.0, 1.50, 0.01)
lg_away = sb.number_input("League avg away goals/match", 0.5, 3.0, 1.20, 0.01)

sb.markdown("---")
sb.subheader("📈 Home team (totals over N matches)")
h_n = sb.number_input("Home matches played", 1, 60, 6, 1)
h_xgf = sb.number_input("Home xG for", 0.0, 150.0, 15.17, 0.01)
h_xga = sb.number_input("Home xG against", 0.0, 150.0, 7.00, 0.01)
h_gf = sb.number_input("Home goals for", 0, 200, 16, 1)
h_ga = sb.number_input("Home goals against", 0, 200, 6, 1)

sb.markdown("---")
sb.subheader("📉 Away team (totals over N matches)")
a_n = sb.number_input("Away matches played", 1, 60, 6, 1)
a_xgf = sb.number_input("Away xG for", 0.0, 150.0, 14.59, 0.01)
a_xga = sb.number_input("Away xG against", 0.0, 150.0, 7.00, 0.01)
a_gf = sb.number_input("Away goals for", 0, 200, 13, 1)
a_ga = sb.number_input("Away goals against", 0, 200, 6, 1)

sb.markdown("---")
sb.subheader("⚙️ Model tuning")
w_xg = sb.slider("Weight of xG vs goals", 0.0, 1.0, 0.6, 0.05)
k = sb.slider("Shrinkage (prior matches)", 0, 30, 10, 1)
rho = sb.slider("Dixon-Coles rho", -0.20, 0.0, -0.08, 0.01)

go = sb.button("🚀 Run Analysis")

# ---------- Main ----------
if go:
    lg_avg = (lg_home + lg_away) / 2

    h_att = shrink(per_match(h_xgf, h_gf, h_n, w_xg), lg_avg, h_n, k) / lg_avg
    h_def = shrink(per_match(h_xga, h_ga, h_n, w_xg), lg_avg, h_n, k) / lg_avg
    a_att = shrink(per_match(a_xgf, a_gf, a_n, w_xg), lg_avg, a_n, k) / lg_avg
    a_def = shrink(per_match(a_xga, a_ga, a_n, w_xg), lg_avg, a_n, k) / lg_avg

    lh = lg_home * h_att * a_def
    la = lg_away * a_att * h_def

    m = score_matrix(lh, la, rho)

    st.markdown(f"### 🏟️ **{home_team}** vs **{away_team}**")
    c1, c2 = st.columns(2)
    c1.metric(f"{home_team} xG expected", f"{lh:.2f}")
    c2.metric(f"{away_team} xG expected", f"{la:.2f}")

    p_home = np.tril(m, -1).sum() * 100
    p_draw = np.trace(m) * 100
    p_away = np.triu(m, 1).sum() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home_team} Win", f"{p_home:.1f}%")
    c2.metric("Draw", f"{p_draw:.1f}%")
    c3.metric(f"{away_team} Win", f"{p_away:.1f}%")

    st.markdown("---")
    st.subheader("🎯 Most Probable Scores")
    flat = [(m[i, j], i, j) for i in range(MAX_G + 1) for j in range(MAX_G + 1)]
    for p, i, j in sorted(flat, reverse=True)[:5]:
        st.write(f"- **{home_team} {i} - {j} {away_team}**: `{p*100:.1f}%`")

    st.markdown("---")
    st.subheader("🛡️ Double Chance")
    st.write(f"🔹 1X: `{p_home + p_draw:.1f}%`  |  X2: `{p_away + p_draw:.1f}%`  |  12: `{p_home + p_away:.1f}%`")

    st.markdown("---")
    st.subheader("⚽ Total Goals & BTTS")
    btts = m[1:, 1:].sum() * 100
    st.write(f"BTTS Yes: `{btts:.1f}%` | No: `{100 - btts:.1f}%`")
    st.info(f"Expected total goals: **{lh + la:.2f}**")

    tot = np.zeros(2 * MAX_G + 1)
    for i in range(MAX_G + 1):
        for j in range(MAX_G + 1):
            tot[i + j] += m[i, j]
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        under = tot[: int(line) + 1].sum() * 100
        st.write(f"- Over **{line}**: `{100 - under:.1f}%` | Under: `{under:.1f}%`")

    st.markdown("---")
    st.subheader("🥅 Team Goals")
    for name, marg in [(home_team, m.sum(axis=1)), (away_team, m.sum(axis=0))]:
        st.markdown(f"**{name}**")
        for line in [0.5, 1.5, 2.5]:
            under = marg[: int(line) + 1].sum() * 100
            st.write(f"- Over **{line}**: `{100 - under:.1f}%` | Under: `{under:.1f}%`")

    st.markdown("---")
    st.caption("Probabilities are model estimates. Compare against bookmaker odds to check calibration.")
else:
    st.info("👈 Enter team stats in the sidebar and click **Run Analysis**.")
