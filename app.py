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
    "Premier League (England)": ("fd", "E0"), "Championship (England)": ("fd", "E1"),
    "La Liga (Spain)": ("fd", "SP1"), "Serie A (Italy)": ("fd", "I1"),
    "Bundesliga (Germany)": ("fd", "D1"), "Ligue 1 (France)": ("fd", "F1"),
    "Eredivisie (Netherlands)": ("fd", "N1"), "Liga Portugal": ("fd", "P1"),
    "Super Lig (Turkey)": ("fd", "T1"), "Belgian Pro League": ("fd", "B1"),
    "Scottish Premiership": ("fd", "SC0"), "Greek Super League": ("fd", "G1"),
    "Norway Eliteserien": ("new", "NOR"), "Sweden Allsvenskan": ("new", "SWE"),
    "Russian Premier League": ("new", "RUS"), "Brazil Serie A": ("new", "BRA"),
    "USA MLS": ("new", "USA"), "Argentina Primera": ("new", "ARG"),
    "Mexico Liga MX": ("new", "MEX"), "Japan J1 League": ("new", "JPN"),
    "Denmark Superliga": ("new", "DNK"), "Austria Bundesliga": ("new", "AUT"),
    "Switzerland Super League": ("new", "SWZ"), "Poland Ekstraklasa": ("new", "POL"),
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
           "FTAG": ["AG", "away_score", "away_goals", "away_goal"],
           "HxG": ["home_xg", "xG_home", "xGH", "HomeXG", "home_xG"],
           "AxG": ["away_xg", "xG_away", "xGAway", "AwayXG", "away_xG"]}
TEMPLATE = ("Date,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST\n"
            "15/08/2025,Team A,Team B,2,1,14,9,6,3\n22/08/2025,Team B,Team C,0,0,8,10,2,4\n")


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


def add_xg(df):
    """Real xG if the file has it; otherwise an xG estimate learned from shots; else goals."""
    df = df.copy()
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    if "HxG" in df and "AxG" in df and num("HxG").notna().mean() > 0.5:
        df["XGH"], df["XGA"] = num("HxG").fillna(df.FTHG), num("AxG").fillna(df.FTAG)
        return df, "real xG from the file"
    if all(c in df for c in ["HS", "AS", "HST", "AST"]):
        hs, aw, hst, ast = num("HS"), num("AS"), num("HST"), num("AST")
        ok = hs.notna() & aw.notna() & hst.notna() & ast.notna()
        if ok.sum() >= 100:
            fh = np.column_stack([hst[ok], (hs - hst)[ok].clip(lower=0)])
            fa = np.column_stack([ast[ok], (aw - ast)[ok].clip(lower=0)])
            coef, *_ = np.linalg.lstsq(np.vstack([fh, fa]),
                                       np.concatenate([df.FTHG[ok], df.FTAG[ok]]), rcond=None)
            coef = np.clip(coef, 0, None)
            if coef.sum() > 0:
                df["XGH"] = (hst * coef[0] + (hs - hst).clip(lower=0) * coef[1]).fillna(df.FTHG)
                df["XGA"] = (ast * coef[0] + (aw - ast).clip(lower=0) * coef[1]).fillna(df.FTAG)
                return df, f"estimated from shots (on target x{coef[0]:.2f}, other shots x{coef[1]:.2f})"
    df["XGH"], df["XGA"] = df.FTHG.astype(float), df.FTAG.astype(float)
    return df, "not available (goals only)"


def half_share(df):
    if "HTHG" in df and "HTAG" in df:
        h, a = pd.to_numeric(df.HTHG, errors="coerce"), pd.to_numeric(df.HTAG, errors="coerce")
        ok = h.notna() & a.notna()
        tot = (df.FTHG + df.FTAG)[ok].sum()
        if ok.sum() >= 100 and tot > 0:
            return float(np.clip((h[ok] + a[ok]).sum() / tot, 0.35, 0.55)), "measured in this league's data"
    return 0.45, "default - no half-time data"


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
def fit_dc(df, xi, wx=0.0):
    """Time-weighted Dixon-Coles (MLE). Scoring target = blend of goals and xG/shots-xG."""
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi, ai = df.HomeTeam.map(idx).values, df.AwayTeam.map(idx).values
    xg_, yg_ = df.FTHG.values, df.FTAG.values
    if wx > 0 and "XGH" in df:
        x = (1 - wx) * xg_ + wx * df.XGH.values
        y = (1 - wx) * yg_ + wx * df.XGA.values
    else:
        x, y = xg_.astype(float), yg_.astype(float)
    hv = 1.0 - (df.Neutral.values.astype(float) if "Neutral" in df else 0.0)
    bw = df.Weight.values if "Weight" in df else 1.0
    w = bw * np.exp(-xi * (df.Date.max() - df.Date).dt.days.values)
    m00, m01 = (xg_ == 0) & (yg_ == 0), (xg_ == 0) & (yg_ == 1)
    m10, m11 = (xg_ == 1) & (yg_ == 0), (xg_ == 1) & (yg_ == 1)

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
def get_model(df, xi, wx):
    return fit_dc(df, xi, wx)


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


def backtest(df, xi, n_test, wx, chunks=3):
    n_test = min(n_test, len(df) // 3)
    names = {"main": "Our model + xG/shots" if wx > 0 else "Our model (goals only)"}
    wxs = {"main": wx}
    if wx > 0:
        names["base"], wxs["base"] = "Our model (goals only)", 0.0
    P, Y, MK = {k: [] for k in names}, [], []
    for s in np.array_split(np.arange(len(df) - n_test, len(df)), chunks):
        models = {k: fit_dc(df.iloc[:s[0]], xi, wxs[k]) for k in names}
        for k in s:
            r = df.iloc[k]
            if any(r.HomeTeam not in md["idx"] or r.AwayTeam not in md["idx"] for md in models.values()):
                continue
            neu = bool(r.get("Neutral", 0))
            for key, md in models.items():
                lam, mu = lambdas(md, r.HomeTeam, r.AwayTeam, neu)
                P[key].append(outcome_probs(score_matrix(lam, mu, md["rho"])))
            Y.append(0 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 2))
            MK.append(market_probs(r))
    if not Y:
        return None
    tr = df.iloc[:len(df) - n_test]
    freq = [(tr.FTHG > tr.FTAG).mean(), (tr.FTHG == tr.FTAG).mean(), (tr.FTHG < tr.FTAG).mean()]
    out = {names[k]: metrics(P[k], Y) for k in names}
    out["Naive (league averages)"] = metrics([freq] * len(Y), Y)
    sub = [i for i, m in enumerate(MK) if m is not None]
    if sub:
        out["Our model (matches with odds)"] = metrics([P["main"][i] for i in sub], [Y[i] for i in sub])
        out["Bookmaker odds"] = metrics([MK[i] for i in sub], [Y[i] for i in sub])
    return out


# ======================= TEAM ANALYSIS =======================
def last_matches(df, team, n):
    d = df[(df.HomeTeam == team) | (df.AwayTeam == team)].tail(n)
    rows = []
    for _, r in d.iterrows():
        home = r.HomeTeam == team
        gf, ga = (r.FTHG, r.FTAG) if home else (r.FTAG, r.FTHG)
        xf, xa = (r.XGH, r.XGA) if home else (r.XGA, r.XGH)
        rows.append({"Date": r.Date.strftime("%d/%m/%y"), "Venue": "H" if home else "A",
                     "Opponent": r.AwayTeam if home else r.HomeTeam, "Score": f"{gf}-{ga}",
                     "xG": f"{xf:.1f}-{xa:.1f}", "Res": "W" if gf > ga else ("D" if gf == ga else "L"),
                     "GF": gf, "GA": ga, "XF": xf, "XA": xa})
    return pd.DataFrame(rows)


def team_panel(df, team, has_xg):
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
    if has_xg:
        c = st.columns(2)
        c[0].metric("xG for / match", f"{d10.XF.mean():.2f}")
        c[1].metric("xG against / match", f"{d10.XA.mean():.2f}")
    st.caption("Last 10 matches: " + " ".join(d10.Res))
    cols = ["Date", "Venue", "Opponent", "Score"] + (["xG"] if has_xg else []) + ["Res"]
    st.dataframe(d10.tail(6)[cols], hide_index=True, use_container_width=True)


# ======================= PREDICTION VIEWS =======================
def render_correct_score(m, home, away):
    flat = sorted(((m[i, j], i, j) for i in range(MAX_G + 1) for j in range(MAX_G + 1)), reverse=True)
    top = flat[:10]
    st.markdown("**Top 10 correct scores**")
    st.dataframe(pd.DataFrame({"Score": [f"{home} {i}-{j} {away}" for _, i, j in top],
                               "Probability %": [round(p * 100, 1) for p, _, _ in top],
                               "Fair odds": [round(1 / p, 1) for p, _, _ in top]}),
                 hide_index=True, use_container_width=True)
    parts = []
    for name, cond in [(f"{home} win", lambda i, j: i > j), ("Draw", lambda i, j: i == j),
                       (f"{away} win", lambda i, j: i < j)]:
        p, i, j = next(t for t in flat if cond(t[1], t[2]))
        parts.append(f"**{name}** → {i}-{j} ({p * 100:.1f}%)")
    st.write("Best score inside each result: " + " | ".join(parts))
    st.caption("Even the best single score is usually 10-15%. Use it as a guide, not a certainty.")
    n = 6
    mx = m[:n, :n].max()
    head = "<tr><th></th>" + "".join(f"<th>{j}</th>" for j in range(n)) + "</tr>"
    body = "".join(
        f"<tr><th>{i}</th>" + "".join(
            f"<td style='background:rgba(231,76,60,{0.08 + 0.85 * m[i, j] / mx:.2f});"
            f"text-align:center;padding:4px 7px'>{m[i, j] * 100:.1f}</td>" for j in range(n)) + "</tr>"
        for i in range(n))
    st.markdown(f"**Score grid (%)** - rows: {home} goals, columns: {away} goals")
    st.markdown(f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:13px'>"
                f"{head}{body}</table></div>", unsafe_allow_html=True)


def render_double_chance(m, home, away):
    ph, pdw, pa = outcome_probs(m)
    opts = {f"1X ({home} or draw)": ph + pdw, f"X2 ({away} or draw)": pa + pdw, "12 (no draw)": ph + pa}
    st.dataframe(pd.DataFrame({"Market": list(opts), "Probability %": [round(v * 100, 1) for v in opts.values()],
                               "Fair odds": [round(1 / v, 2) for v in opts.values()]}),
                 hide_index=True, use_container_width=True)
    best = max(opts, key=opts.get)
    st.success(f"Highest-probability pick: **{best}** - {opts[best] * 100:.1f}% (fair odds {1 / opts[best]:.2f})")
    st.write(f"Draw no bet: **{home}** {ph / (ph + pa) * 100:.1f}% (fair odds {(ph + pa) / ph:.2f}) | "
             f"**{away}** {pa / (ph + pa) * 100:.1f}% (fair odds {(ph + pa) / pa:.2f})")
    st.caption("A safer pick pays less. It is only worth it if the bookmaker's odds are above the fair odds.")


def render_handicap(m, lam, mu, home, away):
    ks = np.arange(-MAX_G, MAX_G + 1)
    dp = np.array([np.trace(m, offset=-int(k)) for k in ks])
    res, rows = {}, []
    for L in [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]:
        r = ks + L
        w_, p_, l_ = dp[r > 1e-9].sum(), dp[np.abs(r) < 1e-9].sum(), dp[r < -1e-9].sum()
        res[L] = (w_, p_, l_)
        rows.append({"Handicap": f"{home} {L:+g}", "Win %": round(w_ * 100, 1), "Refund %": round(p_ * 100, 1),
                     "Lose %": round(l_ * 100, 1), "Fair odds": round(1 + l_ / w_, 2) if w_ > 0 else None})
    st.markdown(f"**Asian handicap** (line added to {home}; the {away} line is the opposite)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    fav = home if lam >= mu else away
    fav_win = lambda k: res[-k][0] if fav == home else res[k][2]
    cands = [k for k in (0.5, 1.5, 2.5) if fav_win(k) >= 0.55]
    if cands:
        k = max(cands)
        st.success(f"Strongest line for the favourite: **{fav} -{k:g}** - {fav_win(k) * 100:.1f}% "
                   f"(fair odds {1 / fav_win(k):.2f})")
    else:
        st.info(f"No half-goal line where {fav} is above 55%: the match looks tight.")
    dog = away if fav == home else home
    st.write(f"Safest cover: **{dog} +1.5** - {(1 - fav_win(1.5)) * 100:.1f}%")
    st.markdown("**European (3-way) handicap**")
    er = []
    for h in (-2, -1, 1, 2):
        r = ks + h
        er.append({"Handicap": f"{home} {h:+d}", "Home %": round(dp[r > 0].sum() * 100, 1),
                   "Draw %": round(dp[r == 0].sum() * 100, 1), "Away %": round(dp[r < 0].sum() * 100, 1)})
    st.dataframe(pd.DataFrame(er), hide_index=True, use_container_width=True)
    st.caption("Refund % = stake returned on whole-number lines. Fair odds assume zero bookmaker margin.")


def render_halves(lam, mu, home, away, s, s_src):
    t, g = lam + mu, np.arange(11)
    p1, p2 = poisson.pmf(g, s * t), poisson.pmf(g, (1 - s) * t)
    p1, p2 = p1 / p1.sum(), p2 / p2.sum()
    joint = np.outer(p1, p2)
    c = st.columns(3)
    c[0].metric("1st half goals", f"{s * t:.2f}")
    c[1].metric("2nd half goals", f"{(1 - s) * t:.2f}")
    c[2].metric("Both halves total", f"{t:.2f}")
    st.caption(f"1st half share of goals: {s * 100:.0f}% ({s_src})")
    rows = [{"Line": ln, "1st half over %": round((1 - p1[:int(ln) + 1].sum()) * 100, 1),
             "2nd half over %": round((1 - p2[:int(ln) + 1].sum()) * 100, 1)} for ln in (0.5, 1.5, 2.5)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.write(f"Half with more goals: 1st **{np.tril(joint, -1).sum() * 100:.1f}%** | "
             f"equal **{np.trace(joint) * 100:.1f}%** | 2nd **{np.triu(joint, 1).sum() * 100:.1f}%**")
    hh, hd, ha = outcome_probs(np.outer(poisson.pmf(g, s * lam), poisson.pmf(g, s * mu)))
    st.write(f"Half-time result: {home} **{hh * 100:.1f}%** | Draw **{hd * 100:.1f}%** | {away} **{ha * 100:.1f}%**")
    st.write(f"Expected goals (1st / 2nd half) - {home}: {s * lam:.2f} / {(1 - s) * lam:.2f} | "
             f"{away}: {s * mu:.2f} / {(1 - s) * mu:.2f}")


def render_goals(m, lam, mu, home, away):
    btts = m[1:, 1:].sum() * 100
    st.write(f"BTTS Yes: `{btts:.1f}%` | No: `{100 - btts:.1f}%` | Expected total: **{lam + mu:.2f}**")
    tot = np.zeros(2 * MAX_G + 1)
    for i in range(MAX_G + 1):
        for j in range(MAX_G + 1):
            tot[i + j] += m[i, j]
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        u = tot[:int(line) + 1].sum() * 100
        st.write(f"- Over **{line}**: `{100 - u:.1f}%` | Under: `{u:.1f}%`")
    st.markdown("**Team goals**")
    for name, marg in [(home, m.sum(axis=1)), (away, m.sum(axis=0))]:
        parts = [f"O{ln}: `{100 - marg[:int(ln) + 1].sum() * 100:.0f}%`" for ln in [0.5, 1.5, 2.5]]
        st.write(f"**{name}** - " + " | ".join(parts))


# ======================= UI =======================
st.set_page_config(page_title="Pro Football AI Engine", layout="wide")
st.title("⚽ Pro Football AI Engine")
st.caption("Time-weighted Dixon-Coles model trained on real results, shots and xG")

sb = st.sidebar
sb.header("Settings")
src = sb.selectbox("Competition", list(SOURCES))
kind, code = SOURCES[src]
years = sb.slider("Years of history", 1, 6, 3)
half_life = sb.slider("Recency half-life (days)", 60, 900, 300, 30)
wx_user = sb.slider("xG / shots weight (0 = goals only)", 0.0, 1.0, 0.5, 0.05)

if kind == "upload":
    sb.download_button("Download CSV template", TEMPLATE, "template.csv", "text/csv")
    up = sb.file_uploader("Upload results CSV", type="csv")
    if up is None:
        st.info("No free automatic data feed for this competition. Upload a CSV with the columns "
                "Date, HomeTeam, AwayTeam, FTHG, FTAG (download the template from the sidebar). "
                "Optional columns that improve accuracy: HS, AS, HST, AST (shots) or HxG, AxG (real xG).")
        st.stop()
    df = clean(normalize(pd.read_csv(up, encoding="latin-1")))
else:
    df = load_data(kind, code, years)

if df.empty:
    st.error("Could not load match data (check the file columns or the host's internet connection).")
    st.stop()
if len(df) < 60:
    st.warning("Very little data: predictions will be unreliable.")

df, xg_src = add_xg(df)
has_xg = not xg_src.startswith("not available")
wx = wx_user if has_xg else 0.0
s_share, s_src = half_share(df)
sb.caption(f"xG source: {xg_src}")

xi = float(np.log(2) / half_life)
model = get_model(df, xi, wx)

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
           f"home advantage ×{np.exp(model['home']):.2f} | xG: {xg_src}")
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
    sub = st.tabs(["🎯 Correct score", "🛡️ Double chance", "⚖️ Handicap", "⏱️ Halves", "⚽ Goals & BTTS"])
    with sub[0]:
        render_correct_score(m, home, away)
    with sub[1]:
        render_double_chance(m, home, away)
    with sub[2]:
        render_handicap(m, lam, mu, home, away)
    with sub[3]:
        render_halves(lam, mu, home, away, s_share, s_src)
    with sub[4]:
        render_goals(m, lam, mu, home, away)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        team_panel(df, home, has_xg)
    with c2:
        team_panel(df, away, has_xg)
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
             "it has never seen. With xG/shots on, it is compared against the goals-only version.")
    n_test = st.slider("Matches to test on", 60, 300, 120, 10)
    if st.button("Run backtest"):
        with st.spinner("Backtesting..."):
            res = backtest(df, xi, n_test, wx)
        if res is None:
            st.warning("Not enough data.")
        else:
            st.dataframe(pd.DataFrame(res).T.round(3), use_container_width=True)
            st.caption("Random guessing scores Brier ≈ 0.667. Lower Brier / log loss is better. "
                       "Bookmakers are usually the toughest benchmark.")

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
