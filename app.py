import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Pro Football AI Engine", layout="centered")

st.title("⚽ Pro Football AI Engine")
st.subheader("📊 Ultimate Match Predictor & Comprehensive Analysis")

st.sidebar.header("Match Data Input (WhoScored)")

# Team Names Selection
home_team = st.sidebar.text_input("Home Team", "Barcelona")
away_team = st.sidebar.text_input("Away Team", "Real Madrid")

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Home Team Statistics")
home_xg = st.sidebar.number_input("Home xG", min_value=0.0, max_value=50.0, value=15.17, step=0.01)
home_goals = st.sidebar.number_input("Home Goals", min_value=0, max_value=100, value=16, step=1)
home_xg_diff = st.sidebar.number_input("Home xGDiff", min_value=-20.0, max_value=20.0, value=0.83, step=0.01)
home_shots = st.sidebar.number_input("Home Shots", min_value=0, max_value=500, value=84, step=1)
home_rating = st.sidebar.number_input("Home Rating", min_value=0.0, max_value=10.0, value=7.28, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("📉 Away Team Statistics")
away_xg = st.sidebar.number_input("Away xG", min_value=0.0, max_value=50.0, value=14.59, step=0.01)
away_goals = st.sidebar.number_input("Away Goals", min_value=0, max_value=100, value=13, step=1)
away_xg_diff = st.sidebar.number_input("Away xGDiff", min_value=-20.0, max_value=20.0, value=-1.59, step=0.01)
away_shots = st.sidebar.number_input("Away Shots", min_value=0, max_value=500, value=104, step=1)
away_rating = st.sidebar.number_input("Away Rating", min_value=0.0, max_value=10.0, value=7.05, step=0.01)

matches_played = st.sidebar.number_input("Matches Played", min_value=1, max_value=38, value=6, step=1)

analyze_button = st.sidebar.button("🚀 Run Comprehensive Analysis")

if analyze_button:
    st.markdown(f"### 🏟️ Match Analysis: **{home_team}** vs **{away_team}**")
    
    # Lambda calculation based on weighted parameters
    h_lambda = ((home_xg / matches_played) * 0.4) + ((home_goals / matches_played) * 0.4) + ((home_rating / 10) * 0.2)
    a_lambda = ((away_xg / matches_played) * 0.4) + ((away_goals / matches_played) * 0.4) + ((away_rating / 10) * 0.2)
    
    # Poisson matrix distribution
    max_goals = 6
    poisson_matrix = np.outer(
        [poisson.pmf(i, h_lambda) for i in range(max_goals + 1)],
        [poisson.pmf(j, a_lambda) for j in range(max_goals + 1)]
    )
    
    # 1. Match Result Probabilities
    home_win_prob = np.sum(np.tril(poisson_matrix, -1)) * 100
    draw_prob = np.sum(np.diag(poisson_matrix)) * 100
    away_win_prob = np.sum(np.triu(poisson_matrix, 1)) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{home_team} Win", f"{home_win_prob:.1f}%")
    col2.metric("Draw", f"{draw_prob:.1f}%")
    col3.metric(f"{away_team} Win", f"{away_win_prob:.1f}%")
    
    st.markdown("---")
    
    # 2. Exact Correct Score Prediction
    max_idx = np.unravel_index(np.argmax(poisson_matrix), poisson_matrix.shape)
    best_home_score, best_away_score = max_idx[0], max_idx[1]
    exact_score_prob = poisson_matrix[max_idx] * 100
    st.subheader("🎯 Most Probable Correct Score")
    st.success(f"Predicted Score Line: **{home_team} {best_home_score} - {best_away_score} {away_team}** (Probability: `{exact_score_prob:.1f}%`)")
    
    st.markdown("---")
    
    # 3. Double Chance Market
    st.subheader("🛡️ Double Chance Market")
    dc_1x = home_win_prob + draw_prob
    dc_x2 = away_win_prob + draw_prob
    dc_12 = home_win_prob + away_win_prob
    st.write(f"🔹 **{home_team} or Draw (1X):** `{dc_1x:.1f}%`")
    st.write(f"🔹 **{away_team} or Draw (X2):** `{dc_x2:.1f}%`")
    st.write(f"🔹 **Either Team to Win (12):** `{dc_12:.1f}%`")
    
    st.markdown("---")
    
    # 4. BTTS & Goals Markets
    st.subheader("⚽ Goals & BTTS Markets")
    btts_prob = (1 - poisson.pmf(0, h_lambda)) * (1 - poisson.pmf(0, a_lambda)) * 100
    if btts_prob > 55:
        st.success(f"✅ **Both Teams to Score (BTTS - Yes):** Strongly Recommended (`{btts_prob:.1f}%`)")
    else:
        st.warning(f"⚠️ **Both Teams to Score (BTTS - No/Uncertain):** Estimated at `{btts_prob:.1f}%`")
        
    total_goals_exp = h_lambda + a_lambda
    st.info(f"📊 Expected Total Match Goals: **{total_goals_exp:.2f}**")
    
    prob_over_15 = (1 - np.sum(poisson_matrix[0, 0] + poisson_matrix[0, 1] + poisson_matrix[1, 0])) * 100
    prob_over_25 = (1 - np.sum([poisson_matrix[i, j] for i in range(4) for j in range(4) if i + j <= 2])) * 100
    st.write(f"- Over **1.5 Goals:** `{prob_over_15:.1f}%` | Over **2.5 Goals:** `{prob_over_25:.1f}%` -")

    st.markdown("---")

    # 5. Winning Margin
    st.subheader("⚖️ Winning Margin Analysis")
    if abs(h_lambda - a_lambda) > 0.6:
        favored = home_team if h_lambda > a_lambda else away_team
        st.info(f"🔥 Expecting **{favored}** to win by a comfortable margin (1+ goals).")
    else:
        st.info("⚖️ Closely contested match expected; likely a tight margin or a draw.")

    # 6. Halves Market Insights
    st.subheader("⏱️ Halves Market Insights")
    st.write("🔹 Statistical trends indicate the second half typically sees higher scoring activity due to fatigue and tactical substitutions.")

else:
    st.info("👈 Please input the team statistics from the sidebar and click **Run Comprehensive Analysis** to display all predictions.")
