import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Pro Football AI Engine", layout="centered")

st.title("⚽ Pro Football AI Engine - Ultimate Predictor")

st.sidebar.header("Match Teams & Stats Input")

# اختيار أسماء الفرق
home_team = st.sidebar.text_input("Home Team", "Barcelona")
away_team = st.sidebar.text_input("Away Team", "Real Madrid")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Home Team Detailed Stats")
home_xg = st.sidebar.number_input("Home xG (Total)", min_value=0.0, max_value=50.0, value=15.17, step=0.1)
home_goals = st.sidebar.number_input("Home Goals", min_value=0, max_value=100, value=16, step=1)
home_shots = st.sidebar.number_input("Home Shots", min_value=0, max_value=500, value=84, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Away Team Detailed Stats")
away_xg = st.sidebar.number_input("Away xG (Total)", min_value=0.0, max_value=50.0, value=14.59, step=0.1)
away_goals = st.sidebar.number_input("Away Goals", min_value=0, max_value=100, value=13, step=1)
away_shots = st.sidebar.number_input("Away Shots", min_value=0, max_value=500, value=104, step=1)

# عدد المباريات لتقسيم المعدل (افتراضياً 6 مباريات مثلاً كمثال أو إدخاله)
matches_played = st.sidebar.number_input("Matches Played (approx)", min_value=1, max_value=38, value=6, step=1)

analyze_button = st.sidebar.button("🚀 Run Comprehensive Analysis")

if analyze_button:
    st.subheader(f"Match Analysis: {home_team} vs {away_team}")
    
    # حساب المتوسطات لكل مباراة بناءً على الإجمالي وعدد الماتشات
    h_lambda = (home_xg / matches_played) * 0.5 + (home_goals / matches_played) * 0.5
    a_lambda = (away_xg / matches_played) * 0.5 + (away_goals / matches_played) * 0.5
    
    # مصفوفة احتمالات بواسون للنتائج (حتى 6 أهداف لكل فريق)
    max_goals = 6
    poisson_matrix = np.outer(
        [poisson.pmf(i, h_lambda) for i in range(max_goals + 1)],
        [poisson.pmf(j, a_lambda) for j in range(max_goals + 1)]
    )
    
    home_win_prob = np.sum(np.tril(poisson_matrix, -1)) * 100
    draw_prob = np.sum(np.diag(poisson_matrix)) * 100
    away_win_prob = np.sum(np.triu(poisson_matrix, 1)) * 100
    
    # عرض النتائج في واجهة مستخدم منسقة
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{home_team} Win", f"{home_win_prob:.1f}%")
    col2.metric("Draw", f"{draw_prob:.1f}%")
    col3.metric(f"{away_team} Win", f"{away_win_prob:.1f}%")
    
    st.markdown("---")
    st.subheader("🎯 Expected Match Goals & Markets")
    st.write(f"**Expected Goals (xG Rate):** {home_team}: `{h_lambda:.2f}` | {away_team}: `{a_lambda:.2f}`")
    
    total_goals_exp = h_lambda + a_lambda
    st.write(f"**Total Expected Goals:** `{total_goals_exp:.2f}`")
    
    # اقتراحات سوق الرهانات البسيطة
    st.markdown("### 💡 Betting Market Insights")
    if total_goals_exp > 2.5:
        st.success("🔥 Over 2.5 Goals: **Recommended** (High scoring expectation)")
    else:
        st.info("🛡️ Under 2.5 Goals: **Recommended** (Tighter tactical match)")
        
    if h_lambda > 1.3 and a_lambda > 1.3:
        st.success("⚽ Both Teams to Score (BTTS): **Yes**")
    else:
        st.warning("⚠️ Both Teams to Score (BTTS): **Uncertain / No**")

else:
    st.info("👈 Please input the team stats from the sidebar and click **Run Comprehensive Analysis** to view the full prediction model.")
