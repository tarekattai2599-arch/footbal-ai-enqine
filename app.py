import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_G = 8
LEAGUES = {
    "Premier League": "E0", "Championship": "E1", "La Liga": "SP1",
    "Bundesliga": "D1", "Serie A": "I1", "Ligue 1": "F1",
    "Eredivisie": "N1", "Liga Portugal": "P1", "Super Lig": "T1",
}
ODDS_SETS = [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA"),
             ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]


# ======================= DATA =======================
def clean(df):
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if any(c not in df.columns for c in need):
        return pd.DataFrame()
    df = df.dropna(subset=need).copy()
    try:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading historical match data...")
def load_data(league, n_seasons):
    today = pd.Timestamp.today()
    y0 = today.year if today.month >= 7 else today.year - 1
    frames = []
    for y in range(y0, y0 - n_seasons, -1):
        code = f"{y % 100:02d}{(y + 1) % 100:02d}"
        url = f"https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"
        try:
            d = pd.read_csv(url, encoding="latin-1", on_bad_lines="skip")
            d["Season"] = f"{y}/{y + 1}"
            frames.append(d)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return clean(pd.concat(frames, ignore_index=True))


# ======================= MODEL =======================
def fit_dc(df, xi):
    """Time-weighted Dixon-Coles fitted by maximum likelihood."""
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = df.HomeTeam.map(idx).values
    ai = df.AwayTeam.map(idx).values
    x, y = df.FTHG.values, df.FTAG.values
    days = (df.Date.max() - df.Date).dt.days.values
    w = np.exp(-xi * days)
    m00, m01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
    m10, m11 = (x == 1) & (y == 0), (x == 1) & (y == 1)

    def unpack(p):
        return p[:n] - p[:n].mean(), p[n:2 * n] - p[n:2 * n].mean(), p[2 * n], p[2 * n + 1]

    def nll(p):
        att, dfn, home, rho = unpack(p)
        el = att[hi] + dfn[ai] + home
        em = att[ai] + dfn[hi]
        lam, mu = np.exp(el), np.exp(em)
        tau = np.ones_like(lam)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        tau = np.clip(tau, 1e-6, None)
        ll = np.log(tau) + x * el - lam + y * em - mu
        return -(w * ll).sum() + 0.05 * (p[:2 * n] ** 2).sum()

    p0 = np.zeros(2 * n + 2)
    p0[2 * n] = 0.25
    bounds = [(-3, 3)] * (2 * n) + [(-0.5, 1.0), (-0.3, 0.3)]
    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
    att, dfn, home, rho = unpack(res.x)
    return {"teams": teams, "idx": idx, "att": att, "dfn": dfn, "home": home, "rho": rho}


@st.cache_data(show_spinner="Training model...")
def get_model(df, xi):
    return fit_dc(df, xi)


def lambdas(model, h, a):
    i, j = model["idx"][h], model["idx"][a]
    lam = np.exp(model["att"][i] + model["dfn"][j] + model["home"])
    mu = np.exp(model["att"][j] + model["dfn"][i])
    return lam, mu


def score_matrix(lam, mu, rho):
    g = np.arange(MAX_G + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def outcome_probs(m):
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()


# ======================= BACKTEST =======================
def market_probs(row):
    for cols in ODDS_SETS:
        if all(c in row.index for c in cols):
            o = np.array([row[c] for c in cols], dtype=float)
            if np.all(np.isfinite(o)) and np.all(o > 1):
                inv = 1 / o
                return inv / inv.sum()
    return None


def metrics(P, y):
    P, y = np.array(P), np.array(y)
    O = np.eye(3)[y]
    return {"Accuracy %": (P.argmax(1) == y).mean() * 100,
            "Brier (lower=better)": ((P - O) ** 2).sum(1).mean(),
            "Log loss (lower=better)": -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean(),
            "Matches": len(y)}


def backtest(df, xi, n_test, chunks=3):
    n_test = min(n_test, len(df) // 3)
    test_idx = np.array_split(np.arange(len(df) - n_test, len(df)), chunks)
    P, Y, MK = [], [], []
    for s in test_idx:
        model = fit_dc(df.iloc[:s[0]], xi)
        for k in s:
            r = df.iloc[k]
            if r.HomeTeam not in model["idx"] or r.AwayTeam not in model["idx"]:
                continue
            lam, mu = lambdas(model, r.HomeTeam, r.AwayTeam)
            P.append(outcome_probs(score_matrix(lam, mu, model["rho"])))
            Y.append(0 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 2))
            MK.append(market_probs(r))
    if not P:
        return None
    train = df.iloc[:len(df) - n_test]
    freq = [(train.FTHG > train.FTAG).mean(), (train.FTHG == train.FTAG).mean(),
            (train.FTHG < train.FTAG).mean()]
    out = {"Our model": metrics(P, Y),
           "Naive (league averages)": metrics([freq] * len(Y), Y)}
    sub = [i for i, m in enumerate(MK) if m is not None]
    if sub:
        out["Our model (matches with odds)"] = metrics([P[i] for i in sub], [Y[i] for i in sub])
        out["Bookmaker odds"] = metrics([MK[i] for i in sub], [Y[i] for i in sub])
    return out


# ======================= ANALYSIS =======================
def last_matches(df, team, n):
    d = df[(df.HomeTeam == team) | (df.AwayTeam == team)].tail(n)
    rows = []
    for _, r in d.iterrows():
        home = r.HomeTeam == team
        gf, ga = (r.FTHG, r.FTAG) if home else (r.FTAG, r.FTHG)
        rows.append({"Date": r.Date.strftime("%d/%m/%y"), "Venue": "H" if home else "A",
                     "Opponent": r.AwayTeam if home else r.HomeTeam,
                     "Score": f"{gf}-{ga}", "Res": "W" if gf > ga else ("D" if gf == ga else "L"),
                     "GF": gf, "GA": ga})
    return pd.DataFrame(rows)


def team_panel(df, team):
    st.markdown(f"**{team}**")
    d10 = last_matches(df, team, 10)
    if d10.empty:
        st.write("No data")
        return
    ppg = ((d10.Res == "W") * 3 + (d10.Res == "D")).mean()
    c = st.columns(3)
    c[0].metric("Goals for / match", f"{d10.GF.mean():.2f}")
    c[1].metric("Goals against / match", f"{d10.GA.mean():.2f}")
    c[2].metric("Points / match", f"{ppg:.2f}")
    st.caption("Last 10 matches: " + " ".join(d10.Res))
    st.dataframe(d10.tail(6)[["Date", "Venue", "Opponent", "Score", "Res"]],
                 hide_index=True, use_container_width=True)


def render_markets(m, lam, mu, home, away):
    st.subheader("🎯 Most probable scores")
    flat = sorted(((m[i, j], i, j) for i in range(MAX_G + 1) for j in range(MAX_G + 1)), reverse=True)[:5]
    for p, i, j in flat:
        st.write(f"- **{home} {i} - {j} {away}**: `{p * 100:.1f}%`")

    ph, pdw, pa = outcome_probs(m)
    st.subheader("🛡️ Double chance")
    st.write(f"1X: `{(ph + pdw) * 100:.1f}%` | X2: `{(pa + pdw) * 100:.1f}%` | 12: `{(ph + pa) * 100:.1f}%`")

    st.subheader("⚽ Goals & BTTS")
    btts = m[1:, 1:].sum() * 100
    st.write(f"BTTS Yes: `{btts:.1f}%` | No: `{100 - btts:.1f}%` | Expected total: **{lam + mu:.2f}**")
    tot = np.zeros(2 * MAX_G + 1)
    for i in range(MAX_G + 1):
        for j in range(MAX_G + 1):
            tot[i + j] += m[i, j]
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        u = tot[:int(line) + 1].sum() * 100
        st.write(f"- Over **{line}**: `{100 - u:.1f}%` | Under: `{u:.1f}%`")

    st.subheader("🥅 Team goals")
    for name, marg in [(home, m.sum(axis=1)), (away, m.sum(axis=0))]:
        parts = []
        for line in [0.5, 1.5, 2.5]:
            u = marg[:int(line) + 1].sum() * 100
            parts.append(f"O{line}: `{100 - u:.0f}%`")
        st.write(f"**{name}** — " + " | ".join(parts))


# ======================= UI =======================
st.set_page_config(page_title="Pro Football AI Engine", layout="wide")
st.title("⚽ Pro Football AI Engine")
st.caption("Time-weighted Dixon-Coles model trained on real historical results")

sb = st.sidebar
sb.header("Settings")
league_name = sb.selectbox("League", list(LEAGUES))
n_seasons = sb.slider("Seasons of history", 1, 6, 3)
half_life = sb.slider("Recency half-life (days)", 60, 900, 300, 30)
upload = sb.file_uploader("Optional: your own football-data CSV", type="csv")

if upload is not None:
    df = clean(pd.read_csv(upload, encoding="latin-1"))
    if not df.empty and "Season" not in df.columns:
        df["Season"] = "upload"
else:
    df = load_data(LEAGUES[league_name], n_seasons)

if df.empty:
    st.error("Could not load match data. Check the internet connection of the host, "
             "or upload a CSV from football-data.co.uk in the sidebar.")
    st.stop()

xi = float(np.log(2) / half_life)
model = get_model(df, xi)

latest = df[df.Season == df.Season.iloc[-1]]
teams = sorted((set(latest.HomeTeam) | set(latest.AwayTeam)) & set(model["teams"]))
sb.markdown("---")
home = sb.selectbox("Home team", teams, index=0)
away = sb.selectbox("Away team", teams, index=min(1, len(teams) - 1))
sb.markdown("**Bookmaker odds (optional)**")
o1 = sb.number_input("Home win odds", 0.0, 100.0, 0.0, 0.01)
ox = sb.number_input("Draw odds", 0.0, 100.0, 0.0, 0.01)
o2 = sb.number_input("Away win odds", 0.0, 100.0, 0.0, 0.01)

st.caption(f"{len(df)} matches | {df.Date.min():%d %b %Y} → {df.Date.max():%d %b %Y} | "
           f"home advantage ×{np.exp(model['home']):.2f}")

if home == away:
    st.warning("Choose two different teams.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📋 Team analysis", "🧪 Backtest", "📈 Ratings"])

with tab1:
    lam, mu = lambdas(model, home, away)
    m = score_matrix(lam, mu, model["rho"])
    ph, pdw, pa = outcome_probs(m)
    st.markdown(f"### 🏟️ {home} vs {away}")
    c = st.columns(2)
    c[0].metric(f"{home} expected goals", f"{lam:.2f}")
    c[1].metric(f"{away} expected goals", f"{mu:.2f}")
    c = st.columns(3)
    c[0].metric(f"{home} win", f"{ph * 100:.1f}%", f"fair odds {1 / ph:.2f}", delta_color="off")
    c[1].metric("Draw", f"{pdw * 100:.1f}%", f"fair odds {1 / pdw:.2f}", delta_color="off")
    c[2].metric(f"{away} win", f"{pa * 100:.1f}%", f"fair odds {1 / pa:.2f}", delta_color="off")

    if o1 > 1 and ox > 1 and o2 > 1:
        inv = 1 / np.array([o1, ox, o2])
        mk = inv / inv.sum()
        mp = np.array([ph, pdw, pa])
        tbl = pd.DataFrame({"Outcome": [f"{home} win", "Draw", f"{away} win"],
                            "Model %": (mp * 100).round(1), "Market %": (mk * 100).round(1),
                            "Edge (pp)": ((mp - mk) * 100).round(1)})
        st.markdown("**Model vs market**")
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        st.caption("A positive edge means the model rates the outcome higher than the bookmaker. "
                   "Treat it as a flag to double-check (injuries, line-ups), not as a sure thing.")
    st.markdown("---")
    render_markets(m, lam, mu, home, away)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        team_panel(df, home)
    with c2:
        team_panel(df, away)
    st.markdown("**Head to head (last 5)**")
    h2h = df[((df.HomeTeam == home) & (df.AwayTeam == away)) |
             ((df.HomeTeam == away) & (df.AwayTeam == home))].tail(5)
    if h2h.empty:
        st.write("No meetings in the loaded data.")
    else:
        h2h = h2h.assign(Score=h2h.FTHG.astype(str) + "-" + h2h.FTAG.astype(str),
                         Date=h2h.Date.dt.strftime("%d/%m/%y"))
        st.dataframe(h2h[["Date", "HomeTeam", "Score", "AwayTeam"]], hide_index=True,
                     use_container_width=True)

with tab3:
    st.write("Walk-forward test: the model is retrained on the past only, then predicts matches "
             "it has never seen. This is the real measure of accuracy.")
    n_test = st.slider("Matches to test on", 60, 300, 120, 10)
    if st.button("Run backtest"):
        with st.spinner("Backtesting..."):
            res = backtest(df, xi, n_test)
        if res is None:
            st.warning("Not enough data.")
        else:
            st.dataframe(pd.DataFrame(res).T.round(3), use_container_width=True)
            st.caption("Random guessing scores Brier ≈ 0.667. Bookmakers are usually the toughest benchmark. "
                       "Football is noisy: 50-55% accuracy on 1X2 is already a strong result.")

with tab4:
    rows = [{"Team": t,
             "Attack (goals vs avg)": round(float(np.exp(model["att"][model["idx"][t]])), 2),
             "Defence (conceded vs avg, lower=better)": round(float(np.exp(model["dfn"][model["idx"][t]])), 2),
             "Overall": round(float(model["att"][model["idx"][t]] - model["dfn"][model["idx"][t]]), 2)}
            for t in teams]
    st.dataframe(pd.DataFrame(rows).sort_values("Overall", ascending=False),
                 hide_index=True, use_container_width=True)

st.caption("Estimates only. The model does not know injuries, line-ups or motivation.")
