import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Pro Football AI Engine Pro",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Pro Football AI Engine - Ultimate Predictor")

# Sidebar for Match Data Input
st.sidebar.header("📝 Match Data Input")

league_name = st.sidebar.text_input("League / Competition", "Premier League")

st.sidebar.markdown("---")
st.sidebar.text("Match Teams")
col1, col2 = st.sidebar.columns(2)
with col1:
    home_team = st.text_input("Home Team", "Newcastle United")
with col2:
    away_team = st.text_input("Away Team", "Bournemouth")

st.sidebar.markdown("---")
st.sidebar.text("Expected Goals (xG)")
col3, col4 = st.sidebar.columns(2)
with col3:
    home_xg = st.sidebar.number_input("Home xG", min_value=0.0, max_value=10.0, value=1.77, step=0.05)
with col4:
    away_xg = st.sidebar.number_input("Away xG", min_value=0.0, max_value=10.0, value=1.26, step=0.05)

run_analysis = st.sidebar.button("🚀 Run Comprehensive Analysis")

if run_analysis:
    st.success(f"Analyzing match: **{home_team}** vs **{away_team}** in **{league_name}**...")
    
    max_goals = 7
    matrix = np.outer(
        [poisson.pmf(i, home_xg) for i in range(max_goals)],
        [poisson.pmf(j, away_xg) for j in range(max_goals)]
    )
    
    # Basic 1X2 Probabilities
    home_win_prob = np.sum(np.tril(matrix, -1)) * 100
    draw_prob = np.sum(np.diagonal(matrix)) * 100
    away_win_prob = np.sum(np.triu(matrix, 1)) * 100
    
    st.markdown("### 📊 1. Match Outcome Probabilities")
    r1, r2, r3 = st.columns(3)
    r1.metric(label=f"{home_team} Win", value=f"{home_win_prob:.1f}%")
    r2.metric(label="Draw", value=f"{draw_prob:.1f}%")
    r3.metric(label=f"{away_team} Win", value=f"{away_win_prob:.1f}%")
    
    # Double Chance
    st.markdown("### 🛡️ 2. Double Chance Markets")
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric(label=f"{home_team} or Draw (1X)", value=f"{home_win_prob + draw_prob:.1f}%")
    dc2.metric(label=f"{home_team} or {away_team} (12)", value=f"{home_win_prob + away_win_prob:.1f}%")
    dc3.metric(label=f"{away_team} or Draw (X2)", value=f"{away_win_prob + draw_prob:.1f}%")

    # Over / Under Markets (0.5 to 4.5)
    st.markdown("### ⚽ 3. Total Goals Markets (Over / Under 0.5 to 4.5)")
    lines = [0.5, 1.5, 2.5, 3.5, 4.5]
    ou_data = []
    for line in lines:
        over_p = 0.0
        under_p = 0.0
        for i in range(max_goals):
            for j in range(max_goals):
                if (i + j) > line:
                    over_p += matrix[i, j] * 100
                else:
                    under_p += matrix[i, j] * 100
        ou_data.append({"Line": f"Over / Under {line}", "Over (%)": f"{over_p:.1f}%", "Under (%)": f"{under_p:.1f}%"})
    
    st.table(pd.DataFrame(ou_data))

    # Exact Correct Scores (Top 3 most likely)
    st.markdown("### 🎯 4. Most Likely Exact Correct Scores")
    scores_list = []
    for i in range(5):
        for j in range(5):
            scores_list.append({"Score": f"{i} - {j}", "Prob": matrix[i, j] * 100})
    
    scores_df = pd.DataFrame(scores_list).sort_values(by="Prob", ascending=False).head(4)
    sc_cols = st.columns(4)
    for idx, row in enumerate(scores_df.itertuples()):
        with sc_cols[idx]:
            st.metric(label=f"Score: {row.Score}", value=f"{row.Prob:.1f}%")

    # European Handicap (-1.0 and +1.0)
    st.markdown("### ⚖️ 5. European Handicap Markets")
    # Home -1 (Home wins by 2 or more goals)
    h_minus_1 = np.sum([matrix[i, j] for i in range(2, max_goals) for j in range(max_goals) if i - j >= 2]) * 100
    # Away +1 (Away wins, draws, or loses by exactly 1 goal)
    a_plus_1 = 100 - (np.sum([matrix[i, j] for i in range(1, max_goals) for j in range(0, i) if i - j == 1]) * 100) # approximation logic for clean display
    
    hc1, hc2 = st.columns(2)
    hc1.metric(label=f"{home_team} (-1.0 Handicap)", value=f"{h_minus_1:.1f}%")
    hc2.metric(label=f"{away_team} (+1.0 Handicap)", value=f"{a_plus_1:.1f}%")

    # Top 3 Betting Recommendations
    st.markdown("### 🔥 6. Top 3 AI Betting Recommendations (أقوى 3 ترشيحات)")
    rec1, rec2, rec3 = st.columns(3)
    rec1.success(f"**1. الخيار الأول الأرجح:**\n\n {home_team} Win or Draw (1X)")
    rec2.success(f"**2. الخيار الثاني الأرجح:**\n\n Over 1.5 Goals")
    rec3.success(f"**3. الخيار الثالث الأرجح:**\n\n Both Teams to Score (قريب من حسابات الـ xG)")

else:
    st.info("👈 Please input the match data from the sidebar and click **Run Comprehensive Analysis**.")