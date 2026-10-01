import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_G = 8
RIDGE = 0.05
FD = "https://www.football-data.co.uk"
INTL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

SOURCES = {
    "Premier League (England)": ("fd", "E0"),
    "Championship (England)": ("fd", "E1"),
    "La Liga (Spain)": ("fd", "SP1"),
    "Serie A (Italy)": ("fd", "I1"),
    "Bundesliga (Germany)": ("fd", "D1"),
    "Ligue 1 (France)": ("fd", "F1"),
    "Eredivisie (Netherlands)": ("fd", "N1"),
    "Liga Portugal": ("fd", "P1"),
    "Super Lig (Turkey)": ("fd", "T1"),
    "Belgian Pro League": ("fd", "B1"),
    "Scottish Premiership": ("fd", "SC0"),
    "Greek Super League": ("fd", "G1"),
    "Norway Eliteserien": ("new", "NOR"),
    "Sweden Allsvenskan": ("new", "SWE"),
    "Russian Premier League": ("new", "RUS"),
    "Brazil Serie A": ("new", "BRA"),
    "USA MLS": ("new", "USA"),
    "Argentina Primera": ("new", "ARG"),
    "Mexico Liga MX": ("new", "MEX"),
    "Japan J1 League": ("new", "JPN"),
    "Denmark Superliga": ("new", "DNK"),
    "Austria Bundesliga": ("new", "AUT"),
    "Switzerland Super League": ("new", "SWZ"),
    "Poland Ekstraklasa": ("new", "POL"),
    "Africa Cup of Nations (national teams)": ("intl", "African Cup of Nations"),
    "UEFA Nations League (national teams)": ("intl", "UEFA Nations League"),
    "Egyptian Premier League (upload CSV)": ("upload", ""),
    "Saudi Pro League (upload CSV)": ("upload", ""),
    "Champions League (upload CSV)": ("upload", ""),
    "Any other competition (upload CSV)": ("upload", ""),
}
ODDS_SETS = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA"),
             ("PH", "PD", "PA"), ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"),
             ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
ALIASES = {"Date": ["date"], "HomeTeam": ["Home", "home_team", "Home Team"],
           "AwayTeam": ["Away", "away_team", "Away Team"],
           "FTHG": ["HG", "home_score", "home_goals", "home_goal"],
           "FTAG": ["AG", "away_score", "away_goals", "away_goal"]}
TEMPLATE = ("Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
            "15/08/2025,Team A,Team B,2,1\n22/08/2025,Team B,Team C,0,0\n")


# ======================= DATA =======================
def normalize(df):
    df = df.copy()
    for std, alts in ALIASES.items():
        if std not in df.columns:
            for a in alts:
                if a in df.columns:
                    df = df.rename(columns={a: std})
                    break
    nc = next((c for c in df.columns if str(c).lower() == "neutral"), None)
    df["Neutral"] = df[nc].astype(str).str.upper().eq("TRUE").astype(int) if nc else 0
    tc = next((c for c in df.columns if str(c).lower() == "tournament"), None)
    df["Tournament"] = df[tc].astype(str) if tc else ""
    df["Weight"] = np.where(df["Tournament"].eq("Friendly"), 0.5, 1.0)
    return df


def clean(df, dayfirst=True):
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if any(c not in df.columns for c in need):
        return pd.DataFrame()
    df = df.dropna(subset=need).copy()
    try:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=dayfirst, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=dayfirst, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading historical match data...")
def load_data(kind, code, years):
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=int(365.25 * years))
    try:
        if kind == "fd":
            y0 = today.year if today.month >= 7 else today.year - 1
            frames = []
            for y in range(y0, y0 - years - 1, -1):
                url = f"{FD}/mmz4281/{y % 100:02d}{(y + 1) % 100:02d}/{code}.csv"
                try:
                    frames.append(pd.read_csv(url, encoding="latin-1", on_bad_lines="skip"))
                except Exception:
                    continue
            df = clean(normalize(pd.concat(frames, ignore_index=True)))
        elif kind == "new":
            df = clean(normalize(pd.read_csv(f"{FD}/new/{code}.csv", encoding="latin-1", on_bad_lines="skip")))
        else:
            df = clean(normalize(pd.read_csv(INTL_URL)), dayfirst=False)
    except Exception:
        return pd.DataFrame()
    return df[df.Date >= cutoff].reset_index(drop=True)


def current_teams(df, min_gap=40):
    d = np.sort(df.Date.dt.normalize().unique())
    gaps = np.diff(d).astype("timedelta64[D]").astype(int)
    big = np.where(gaps > min_gap)[0]
    cur = df[df.Date >= d[big[-1] + 1]] if len(big) else df
    if cur.empty:
        cur = df
    return set(cur.HomeTeam) | set(cur.AwayTeam)


def eligible_teams(df, model, min_matches, recent_days):
    rec = df[df.Date >= df.Date.max() - pd.Timedelta(days=recent_days)]
    rec_t = set(rec.HomeTeam) | set(rec.AwayTeam)
    cnt = pd.concat([df.HomeTeam, df.AwayTeam]).value_counts()
    return sorted(t for t in model["teams"] if t in rec_t and cnt.get(t, 0) >= min_matches)


# ======================= MODEL =======================
def fit_dc(df, xi):
    """Time-weighted Dixon-Coles (MLE, analytic gradient, neutral-venue aware)."""
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi, ai = df.HomeTeam.map(idx).values, df.AwayTeam.map(idx).values
    x, y = df.FTHG.values, df.FTAG.values
    hv = 1.0 - (df.Neutral.values.astype(float) if "Neutral" in df else 0.0)
    bw = df.Weight.values if "Weight" in df else 1.0
    w = bw * np.exp(-xi * (df.Date.max() - df.Date).dt.days.values)
    m00, m01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
    m10, m11 = (x == 1) & (y == 0), (x == 1) & (y == 1)

    def unpack(p):
        return p[:n] - p[:n].mean(), p[n:2 * n] - p[n:2 * n].mean(), p[2 * n], p[2 * n + 1]

    def fun(p):
        att, dfn, home, rho = unpack(p)
        el = att[hi] + dfn[ai] + home * hv
        em = att[ai] + dfn[hi]
        lam, mu = np.exp(el), np.exp(em)
        tau, dtl, dtm, dtr = np.ones_like(lam), np.zeros_like(lam), np.zeros_like(lam), np.zeros_like(lam)
        lm = lam * mu
        tau[m00] = 1 - lm[m00] * rho
        dtl[m00] = dtm[m00] = -lm[m00] * rho
        dtr[m00] = -lm[m00]
        tau[m01] = 1 + lam[m01] * rho
        dtl[m01] = lam[m01] * rho
        dtr[m01] = lam[m01]
        tau[m10] = 1 + mu[m10] * rho
        dtm[m10] = mu[m10] * rho
        dtr[m10] = mu[m10]
        tau[m11] = 1 - rho
        dtr[m11] = -1.0
        tau = np.clip(tau, 1e-6, None)
        ll = np.log(tau) + x * el - lam + y * em - mu
        f = -(w * ll).sum() + RIDGE * (p[:2 * n] ** 2).sum()
        gl = w * (x - lam + dtl / tau)
        gm = w * (y - mu + dtm / tau)
        ga = np.bincount(hi, gl, n) + np.bincount(ai, gm, n)
        gd = np.bincount(ai, gl, n) + np.bincount(hi, gm, n)
        g = np.concatenate([ga - ga.mean(), gd - gd.mean(), [(gl * hv).sum(), (w * dtr / tau).sum()]])
        g = -g
        g[:2 * n] += 2 * RIDGE * p[:2 * n]
        return f, g

    p0 = np.zeros(2 * n + 2)
    p0[2 * n] = 0.25
    bounds = [(-3, 3)] * (2 * n) + [(-0.5, 1.0), (-0.3, 0.3)]
    res = minimize(fun, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    att, dfn, home, rho = unpack(res.x)
    return {"teams": teams, "idx": idx, "att": att, "dfn": dfn, "home": home, "rho": rho}


@st.cache_data(show_spinner="Training model...")
def get_model(df, xi):
    return fit_dc(df, xi)


def lambdas(model, h, a, neutral=False):
    i, j = model["idx"][h], model["idx"][a]
    hadv = 0.0 if neutral else model["home"]
    return (np.exp(model["att"][i] + model["dfn"][j] + hadv),
            np.exp(model["att"][j] + model["dfn"][i]))


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
    P, Y, MK = [], [], []
    for s in np.array_split(np.arange(len(df) - n_test, len(df)), chunks):
        model = fit_dc(df.iloc[:s[0]], xi)
        for k in s:
            r = df.iloc[k]
            if r.HomeTeam not in model["idx"] or r.AwayTeam not in model["idx"]:
                continue
            lam, mu = lambdas(model, r.HomeTeam, r.AwayTeam, bool(r.get("Neutral", 0)))
            P.append(outcome_probs(score_matrix(lam, mu, model["rho"])))
            Y.append(0 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 2))
            MK.append(market_probs(r))
    if not P:
        return None
    tr = df.iloc[:len(df) - n_test]
    freq = [(tr.FTHG > tr.FTAG).mean(), (tr.FTHG == tr.FTAG).mean(), (tr.FTHG < tr.FTAG).mean()]
    out = {"Our model": metrics(P, Y), "Naive (league averages)": metrics([freq] * len(Y), Y)}
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
                     "Opponent": r.AwayTeam if home else r.HomeTeam, "Score": f"{gf}-{ga}",
                     "Res": "W" if gf > ga else ("D" if gf == ga else "L"), "GF": gf, "GA": ga})
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
        parts = [f"O{ln}: `{100 - marg[:int(ln) + 1].sum() * 100:.0f}%`" for ln in [0.5, 1.5, 2.5]]
        st.write(f"**{name}** — " + " | ".join(parts))


# ======================= UI =======================
st.set_page_config(page_title="Pro Football AI Engine", layout="wide")
st.title("⚽ Pro Football AI Engine")
st.caption("Time-weighted Dixon-Coles model trained on real historical results")

sb = st.sidebar
sb.header("Settings")
src = sb.selectbox("Competition", list(SOURCES))
kind, code = SOURCES[src]
years = sb.slider("Years of history", 1, 6, 3)
half_life = sb.slider("Recency half-life (days)", 60, 900, 300, 30)

if kind == "upload":
    sb.download_button("Download CSV template", TEMPLATE, "template.csv", "text/csv")
    up = sb.file_uploader("Upload results CSV", type="csv")
    if up is None:
        st.info("No free automatic data feed for this competition. Upload a CSV with the columns "
                "Date, HomeTeam, AwayTeam, FTHG, FTAG (download the template from the sidebar). "
                "The more past matches, the better the model.")
        st.stop()
    df = clean(normalize(pd.read_csv(up, encoding="latin-1")))
else:
    df = load_data(kind, code, years)

if df.empty:
    st.error("Could not load match data (check the file columns or the host's internet connection).")
    st.stop()
if len(df) < 60:
    st.warning("Very little data: predictions will be unreliable.")

xi = float(np.log(2) / half_life)
model = get_model(df, xi)

if kind == "intl":
    pool_df = df[df.Tournament.str.contains(code, case=False, na=False)]
    pool = set(pool_df.HomeTeam) | set(pool_df.AwayTeam)
    teams = [t for t in eligible_teams(df, model, 6, 730) if t in pool] or eligible_teams(df, model, 6, 730)
else:
    teams = sorted(current_teams(df) & set(model["teams"]))
if len(teams) < 2:
    st.error("Not enough teams found in the data.")
    st.stop()

sb.markdown("---")
home = sb.selectbox("Home team", teams, index=0)
away = sb.selectbox("Away team", teams, index=1)
neutral = sb.checkbox("Neutral venue (no home advantage)", value=(kind == "intl"), key=f"neu_{src}")
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
    lam, mu = lambdas(model, home, away, neutral)
    m = score_matrix(lam, mu, model["rho"])
    ph, pdw, pa = outcome_probs(m)
    st.markdown(f"### 🏟️ {home} vs {away}" + (" (neutral venue)" if neutral else ""))
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
        st.markdown("**Model vs market**")
        st.dataframe(pd.DataFrame({"Outcome": [f"{home} win", "Draw", f"{away} win"],
                                   "Model %": (mp * 100).round(1), "Market %": (mk * 100).round(1),
                                   "Edge (pp)": ((mp - mk) * 100).round(1)}),
                     hide_index=True, use_container_width=True)
        st.caption("A positive edge means the model rates the outcome higher than the bookmaker. "
                   "Double-check injuries and line-ups before trusting it.")
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
                       "50-55% accuracy on 1X2 is already a strong result.")

with tab4:
    rows = []
    for t in teams:
        a, d = model["att"][model["idx"][t]], model["dfn"][model["idx"][t]]
        rows.append({"Team": t, "Attack (goals vs avg)": round(float(np.exp(a)), 2),
                     "Defence (conceded vs avg, lower=better)": round(float(np.exp(d)), 2),
                     "Overall": round(float(a - d), 2)})
    st.dataframe(pd.DataFrame(rows).sort_values("Overall", ascending=False),
                 hide_index=True, use_container_width=True)

st.caption("Estimates only. The model does not know injuries, line-ups or motivation.")
# END OF FILE
