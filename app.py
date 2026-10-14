import datetime as dt
import difflib
import re
import sqlite3

import numpy as np
import pandas as pd
import requests
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



SEL = ["1", "X", "2", "1X", "X2", "12", "Over 1.5", "Over 2.5", "Under 2.5", "Under 3.5", "BTTS Yes", "BTTS No"]
N_SIMS = 20000


# ======================= SLIP ENGINE =======================
def masks(n):
    i, j = np.indices((n, n))
    t = i + j
    return {"1": i > j, "X": i == j, "2": i < j, "1X": i >= j, "X2": i <= j, "12": i != j,
            "Over 1.5": t > 1, "Over 2.5": t > 2, "Under 2.5": t < 3, "Under 3.5": t < 4,
            "BTTS Yes": (i > 0) & (j > 0), "BTTS No": (i == 0) | (j == 0)}


def slip_stats(legs):
    P = float(np.prod([l["p"] for l in legs]))
    O = float(np.prod([l["odds"] or l["fair"] for l in legs]))
    real = all(l["odds"] for l in legs)
    ev = P * O - 1 if real else None
    kelly = max(0.0, (P * O - 1) / (O - 1)) if real and O > 1 else 0.0
    return {"P": P, "O": O, "real_odds": real, "ev": ev, "kelly": kelly, "fair": 1 / P}


def simulate(fx, legs, n_sims=N_SIMS, seed=7):
    """Monte Carlo: sample a scoreline for every match and check which legs win."""
    rng = np.random.default_rng(seed)
    wins = np.zeros((n_sims, len(legs)), dtype=bool)
    for c, l in enumerate(legs):
        m = fx[l["fi"]]["m"]
        n = m.shape[0]
        idx = rng.choice(n * n, size=n_sims, p=(m / m.sum()).ravel())
        i, j = divmod(idx, n)
        wins[:, c] = masks(n)[l["sel"]][i, j]
    lost = (~wins).sum(axis=1)
    return np.bincount(lost, minlength=len(legs) + 1) / n_sims


def map_players(raw):
    out = pd.DataFrame(index=raw.index)
    for std, alts in PALIAS.items():
        col = next((a for a in alts if a in raw.columns), None)
        out[std] = raw[col] if col else np.nan
    for c in out.columns.difference(["player", "team", "pos"]):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["team"] = out["team"].fillna("-")
    out["pos"] = out["pos"].fillna("").astype(str)
    return out


def player_table(d, min_minutes, role, teams_sel):
    tt = (d.npxg.fillna(0) + d.xa.fillna(0)).groupby(d.team).transform("sum")
    d = d.assign(share=(d.npxg.fillna(0) + d.xa.fillna(0)) / tt.replace(0, np.nan) * 100)
    d = d[d.minutes >= min_minutes]
    if teams_sel:
        d = d[d.team.isin(teams_sel)]
    pat = {"Forwards": r"\bF\b", "Midfielders": r"\bM\b", "Defenders": r"\bD\b"}.get(role)
    if pat:
        d = d[d.pos.str.contains(pat, regex=True)]
    p90 = lambda c: d[c] / d.minutes * 90
    t = pd.DataFrame({"Player": d.player, "Team": d.team, "Pos": d.pos, "Min": d.minutes,
                      "Goals": d.goals, "xG": d.xg, "Goals-xG": d.goals - d.xg,
                      "npxG/90": p90("npxg"), "xA/90": p90("xa"), "Shots/90": p90("shots"),
                      "KeyPass/90": p90("kp"), "xGChain/90": p90("chain"), "xGBuildup/90": p90("build"),
                      "Team share %": d.share})
    t["Score"] = (t["xGBuildup/90"] + t["xGChain/90"]) if role == "Defenders" else (t["npxG/90"] + t["xA/90"])
    return t.sort_values("Score", ascending=False).round(2)


# ======================= PLAYERS TAB =======================
def render_players_tab():
    st.write("Upload the players CSV created by `fetch_understat.py` (Understat season stats).")
    up = st.file_uploader("Players CSV", type="csv", key="players_up")
    if up is None:
        st.info("Free sources only give attacking and build-up data (xG, xA, shots, key passes, xGChain, xGBuildup). "
                "Tackles, interceptions and counter-attack data are not available for free since FBref removed its "
                "advanced stats in January 2026.")
        return
    d = map_players(pd.read_csv(up))
    c = st.columns(3)
    role = c[0].radio("Role", ["All", "Forwards", "Midfielders", "Defenders"], key="pl_role")
    mins = c[1].slider("Min minutes played", 0, 3000, 450, 50, key="pl_min")
    tsel = c[2].multiselect("Teams", sorted(d.team.unique()), key="pl_teams")
    t = player_table(d, mins, role, tsel)
    st.dataframe(t, hide_index=True, use_container_width=True)
    st.caption("Score = npxG/90 + xA/90 (defenders: xGChain/90 + xGBuildup/90, in-possession contribution only). "
               "'Team share %' is the player's share of his team's npxG + xA: a high number means the team's attack "
               "depends on him, so his absence matters more.")



# ======================= MATCH BOARD (database + multi-league analysis) =======================
DB_PATH = "football_board.db"
BOARD_LEAGUES = [k for k, (kd, _) in SOURCES.items() if kd in ("fd", "new", "intl")]
FD_CODE_TO_LEAGUE = {code: name for name, (kd, code) in SOURCES.items() if kd == "fd"}
OFB = "https://raw.githubusercontent.com/openfootball/football.json/master"
OF = {"E0": "en.1", "E1": "en.2", "SP1": "es.1", "I1": "it.1", "D1": "de.1", "F1": "fr.1", "N1": "nl.1",
      "P1": "pt.1", "T1": "tr.1", "B1": "be.1", "SC0": "sco.1", "G1": "gr.1"}
FX_ODDS = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
DEFAULT_MARKETS = ["1", "X", "2", "1X", "X2", "12", "Over 1.5", "Over 2.5", "Under 3.5", "BTTS Yes", "BTTS No"]


# ---------- database ----------
def _con():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("CREATE TABLE IF NOT EXISTS fixtures(id INTEGER PRIMARY KEY AUTOINCREMENT, league TEXT, home TEXT, "
                "away TEXT, kickoff TEXT, neutral INTEGER DEFAULT 0, o1 REAL, ox REAL, o2 REAL, "
                "adj_h REAL DEFAULT 1, adj_a REAL DEFAULT 1, note TEXT DEFAULT '', created TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS picks(fixture_id INTEGER, sel TEXT, p REAL, created TEXT, "
                "PRIMARY KEY(fixture_id, sel))")
    return con


def db_read(sql, params=()):
    con = _con()
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def db_exec(sql, params=()):
    con = _con()
    try:
        with con:
            cur = con.execute(sql, params)
        return cur.lastrowid
    finally:
        con.close()


def db_add_fixture(league, home, away, kickoff, neutral=0, o1=None, ox=None, o2=None):
    dup = db_read("SELECT id FROM fixtures WHERE league=? AND home=? AND away=? AND kickoff=?",
                  (league, home, away, str(kickoff)))
    if len(dup):
        return False
    nz = lambda v: float(v) if v is not None and pd.notna(v) and float(v) > 1 else None
    db_exec("INSERT INTO fixtures(league,home,away,kickoff,neutral,o1,ox,o2,created) VALUES(?,?,?,?,?,?,?,?,?)",
            (league, home, away, str(kickoff), int(neutral), nz(o1), nz(ox), nz(o2), dt.datetime.now().isoformat()))
    return True


# ---------- team-name matching and source fusion ----------
SYN = [(r"\bman\b", "manchester"), (r"\butd\b", "united"), (r"\bnott m\b", "nottingham"), (r"\bwolves\b", "wolverhampton wanderers"),
       (r"\bsheff\b", "sheffield"), (r"\bath\b", "athletic"), (r"\bspurs\b", "tottenham hotspur"),
       (r"\bbrighton hove albion\b", "brighton"), (r"\batletico\b", "atletico"), (r"\binternazionale\b", "inter"),
       (r"\bbayern munchen\b", "bayern munich"), (r"\bparis saint germain\b", "paris sg"), (r"\bpsg\b", "paris sg"),
       (r"\bm gladbach\b", "monchengladbach"), (r"\bmgladbach\b", "monchengladbach")]


def _key(s):
    s = re.sub(r"\b(fc|cf|afc|sc|ac|as|cd|sk|if|fk|bk|ik|the)\b", " ", str(s).lower())
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    for pat, rep in SYN:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\b(wanderers|hotspur|albion|hove|city|united)\b", lambda m: m.group(1), s)
    return s


def match_team(name, teams):
    """Map a team name from any source onto the names used by the model. Returns None if nothing close enough."""
    teams = list(teams)
    if name in teams:
        return name
    keys = {_key(t): t for t in teams}
    k = _key(name)
    if k in keys:
        return keys[k]
    cand = [t for kk, t in keys.items() if min(len(k), len(kk)) >= 4 and (k in kk or kk in k)]
    if len(cand) == 1:
        return cand[0]
    best = difflib.get_close_matches(k, list(keys), n=1, cutoff=0.74)
    return keys[best[0]] if best else None


def fuse(primary, others):
    """Merge result tables from several sources. Team names of every extra source are mapped onto the primary
    names; a match already in the primary (same teams, date within one day) is kept as is, and only matches the
    primary does not have are added. Returns (merged_df, number_of_added_matches)."""
    names = set(primary.HomeTeam) | set(primary.AwayTeam)
    known = set()
    for r in primary[["Date", "HomeTeam", "AwayTeam"]].itertuples(index=False):
        d = r.Date.normalize()
        for off in (-1, 0, 1):
            known.add((d + pd.Timedelta(days=off), r.HomeTeam, r.AwayTeam))
    cache, add = {}, []
    for o in others:
        if o is None or o.empty:
            continue
        for r in o.itertuples(index=False):
            h = cache.setdefault(r.HomeTeam, match_team(r.HomeTeam, names))
            a = cache.setdefault(r.AwayTeam, match_team(r.AwayTeam, names))
            if h is None or a is None:
                continue
            key = (r.Date.normalize(), h, a)
            if key in known:
                continue
            known.add(key)
            add.append({"Date": r.Date, "HomeTeam": h, "AwayTeam": a, "FTHG": r.FTHG, "FTAG": r.FTAG,
                        "Neutral": 0, "Tournament": "", "Weight": 1.0})
    if not add:
        return primary, 0
    out = pd.concat([primary, pd.DataFrame(add)], ignore_index=True).sort_values("Date").reset_index(drop=True)
    return out, len(add)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_openfootball(code, years):
    """Second free source (openfootball on GitHub): results and, when published, upcoming matches."""
    f = OF.get(code)
    if not f:
        return pd.DataFrame(), pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    y0 = today.year if today.month >= 7 else today.year - 1
    res, fut = [], []
    for y in range(y0, max(y0 - years, y0 - 3) - 1, -1):
        try:
            r = requests.get(f"{OFB}/{y}-{(y + 1) % 100:02d}/{f}.json", timeout=20)
            r.raise_for_status()
            matches = r.json().get("matches", [])
        except Exception:
            continue
        for m in matches:
            sc = m.get("score")
            ft = sc.get("ft") if isinstance(sc, dict) else sc if isinstance(sc, list) else None
            row = {"Date": m.get("date"), "HomeTeam": m.get("team1"), "AwayTeam": m.get("team2")}
            if isinstance(ft, list) and len(ft) == 2 and all(isinstance(v, (int, float)) for v in ft):
                res.append({**row, "FTHG": ft[0], "FTAG": ft[1]})
            else:
                fut.append(row)
    rdf = clean(normalize(pd.DataFrame(res)), dayfirst=False) if res else pd.DataFrame()
    fdf = pd.DataFrame(fut)
    if len(fdf):
        fdf["Date"] = pd.to_datetime(fdf["Date"], errors="coerce")
        fdf = fdf.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
        fdf = fdf[fdf.Date >= today]
    return rdf, fdf


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def load_fd_fixtures():
    """Upcoming matches (with bookmaker odds when present) for the main leagues from football-data.co.uk."""
    try:
        d = pd.read_csv(f"{FD}/fixtures.csv", encoding="latin-1", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()
    if not {"Div", "Date", "HomeTeam", "AwayTeam"} <= set(d.columns):
        return pd.DataFrame()
    d = d.dropna(subset=["Div", "Date", "HomeTeam", "AwayTeam"]).copy()
    d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed", errors="coerce")
    d = d.dropna(subset=["Date"])
    d["League"] = d["Div"].map(FD_CODE_TO_LEAGUE)
    d = d.dropna(subset=["League"])
    for c1, c2, c3 in FX_ODDS:
        if {c1, c2, c3} <= set(d.columns):
            d["o1"], d["ox"], d["o2"] = d[c1], d[c2], d[c3]
            break
    else:
        d["o1"] = d["ox"] = d["o2"] = np.nan
    return d[["League", "Date", "HomeTeam", "AwayTeam", "o1", "ox", "o2"]].reset_index(drop=True)


@st.cache_data(ttl=6 * 3600, show_spinner="Loading league and training its model...")
def league_bundle(src, years, half_life, wx_user, fuse_on):
    kind, code = SOURCES[src]
    df = load_data(kind, code, years)
    if df.empty:
        return None
    fut, added = pd.DataFrame(), 0
    if kind == "fd" and fuse_on:
        extra, fut = load_openfootball(code, years)
        if not extra.empty:
            df, added = fuse(df, [extra])
    df, xg_src = add_xg(df)
    has_xg = not xg_src.startswith("not available")
    model = fit_dc(df, float(np.log(2) / half_life), wx_user if has_xg else 0.0)
    if kind == "intl":
        pool_df = df[df.Tournament.str.contains(code, case=False, na=False)]
        pool = set(pool_df.HomeTeam) | set(pool_df.AwayTeam)
        teams = [t for t in eligible_teams(df, model, 6, 730) if t in pool] or eligible_teams(df, model, 6, 730)
    else:
        teams = sorted(current_teams(df) & set(model["teams"]))
    s_share, s_src = half_share(df)
    return {"df": df, "model": model, "teams": teams, "xg": xg_src, "fut": fut, "added": added,
            "s_share": s_share, "s_src": s_src, "kind": kind}


# ---------- analysis ----------
def data_quality(b, h, a):
    d = b["df"]
    r = d[d.Date >= d.Date.max() - pd.Timedelta(days=400)]
    n = lambda t: int(((r.HomeTeam == t) | (r.AwayTeam == t)).sum())
    nh, na = n(h), n(a)
    hi, lo = (12, 6) if b["kind"] == "intl" else (30, 18)
    g = "A" if min(nh, na) >= hi else "B" if min(nh, na) >= lo else "C"
    return nh, na, g


def sel_hit(sel, hg, ag):
    t = hg + ag
    return bool({"1": hg > ag, "X": hg == ag, "2": hg < ag, "1X": hg >= ag, "X2": hg <= ag, "12": hg != ag,
                 "Over 1.5": t > 1, "Over 2.5": t > 2, "Under 2.5": t < 3, "Under 3.5": t < 4,
                 "BTTS Yes": hg > 0 and ag > 0, "BTTS No": hg == 0 or ag == 0}[sel])


def analyse_fixture(r, b):
    """Full model output for one stored fixture. Returns None if a team is unknown to the model."""
    model = b["model"]
    if r.home not in model["idx"] or r.away not in model["idx"]:
        return None
    lam, mu = lambdas(model, r.home, r.away, bool(r.neutral))
    lam, mu = lam * float(r.adj_h or 1), mu * float(r.adj_a or 1)
    m = score_matrix(lam, mu, model["rho"])
    mk = masks(m.shape[0])
    market = {"1": r.o1, "X": r.ox, "2": r.o2}
    sels = []
    for k in SEL:
        p = float(m[mk[k]].sum())
        o = market.get(k)
        o = float(o) if o is not None and pd.notna(o) and float(o) > 1 else None
        sels.append({"sel": k, "p": p, "odds": o, "fair": 1 / p if p > 0 else np.inf, "ev": p * o - 1 if o else None})
    flat = np.dstack(np.unravel_index(np.argsort(-m.ravel())[:5], m.shape))[0]
    tops = [(int(i), int(j), float(m[i, j])) for i, j in flat]
    nh, na, grade = data_quality(b, r.home, r.away)
    return {"id": int(r.id), "league": r.league, "home": r.home, "away": r.away, "kickoff": r.kickoff,
            "lam": lam, "mu": mu, "m": m, "sels": sels, "tops": tops, "nh": nh, "na": na, "grade": grade,
            "adj": (float(r.adj_h or 1), float(r.adj_a or 1)), "note": r.note or "", "b": b}


def best_pick(f, allowed, min_p, min_fair):
    c = [s for s in f["sels"] if s["sel"] in allowed and s["p"] >= min_p and s["fair"] >= min_fair]
    return max(c, key=lambda s: s["p"]) if c else None


def explain(f):
    P = {s["sel"]: s["p"] for s in f["sels"]}
    h, a = f["home"], f["away"]
    top = max(("1", "X", "2"), key=lambda k: P[k])
    if P[top] >= 0.45:
        who = {"1": h, "X": "a draw", "2": a}[top]
        s = f"{who} is the most likely result ({P[top] * 100:.0f}%)."
    else:
        s = "Open match with no clear favourite."
    s += f" Expected goals {f['lam']:.2f}-{f['mu']:.2f}; most likely score {f['tops'][0][0]}-{f['tops'][0][1]}."
    s += f" Over 2.5 goals {P['Over 2.5'] * 100:.0f}%, both teams to score {P['BTTS Yes'] * 100:.0f}%."
    if f["grade"] == "C":
        s += f" Limited recent data ({f['nh']} and {f['na']} matches in the last 400 days): treat with caution."
    if f["adj"] != (1.0, 1.0):
        s += f" Manual adjustment applied (x{f['adj'][0]:.2f} / x{f['adj'][1]:.2f})."
    ev = [(x["sel"], x["ev"]) for x in f["sels"] if x["ev"] is not None]
    if ev:
        k, v = max(ev, key=lambda t: t[1])
        s += f" Versus the market the biggest gap is on '{k}' ({v * 100:+.0f}% value)."
    return s


def top_legs(fxs, allowed, min_p, min_fair):
    legs = []
    for fi, f in enumerate(fxs):
        s = best_pick(f, allowed, min_p, min_fair)
        if s:
            legs.append({**s, "match": f"{f['home']} vs {f['away']}", "fi": fi, "league": f["league"], "score": s["p"]})
    return sorted(legs, key=lambda l: -l["score"])


def save_picks(fxs, allowed, min_p, min_fair):
    """Remember the pre-match picks so they can be graded later. Never overwrites a match that already kicked off."""
    today = dt.date.today().isoformat()
    for f in fxs:
        if f["kickoff"] and f["kickoff"] < today:
            continue
        chosen = {max(("1", "X", "2"), key=lambda k: next(s["p"] for s in f["sels"] if s["sel"] == k))}
        bp = best_pick(f, allowed, min_p, min_fair)
        if bp:
            chosen.add(bp["sel"])
        for k in chosen:
            p = next(s["p"] for s in f["sels"] if s["sel"] == k)
            db_exec("INSERT OR REPLACE INTO picks(fixture_id,sel,p,created) VALUES(?,?,?,?)",
                    (f["id"], k, float(p), dt.datetime.now().isoformat()))


# ---------- AI web research (optional, needs an Anthropic API key) ----------
def ai_research(api_key, model_id, f, max_uses):
    prompt = (f"Research the football match {f['home']} vs {f['away']} ({f['league']}, {f['kickoff']}). "
              "Search the web for the latest information: confirmed injuries and suspensions for both teams, "
              "expected or confirmed line-ups, recent form (last 5 results), and context that changes motivation or "
              "rotation (cup games, congested schedule, nothing to play for). Use reliable recent sources, give the "
              "date of each source, and say clearly when something is unconfirmed. Finish with a short list per team: "
              "key absentees (name and role) and whether this should push that team's expected goals up or down, and "
              "roughly by how much. Keep it under 300 words.")
    body = {"model": model_id, "max_tokens": 1500, "messages": [{"role": "user", "content": prompt}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": int(max_uses)}]}
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=180, json=body,
                          headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"})
    except Exception as e:
        return None, {}, f"Network error: {e}"
    if r.status_code != 200:
        return None, {}, f"API error {r.status_code}: {r.text[:300]}"
    text, cites = [], {}
    for blk in r.json().get("content", []):
        if blk.get("type") == "text":
            text.append(blk.get("text", ""))
            for c in blk.get("citations") or []:
                if c.get("url"):
                    cites[c["url"]] = c.get("title") or c["url"]
    return "\n".join(text).strip(), cites, None


# ---------- UI pieces ----------
def ui_add_fixtures(years, half_life, wx_user, fuse_on):
    st.markdown("#### Add matches")
    c = st.columns([2, 1, 1])
    league = c[0].selectbox("League", BOARD_LEAGUES, key="fx_league")
    kick = c[1].date_input("Match date", value=dt.date.today(), key="fx_date")
    neutral = c[2].checkbox("Neutral venue", value=league.endswith("(national teams)"), key=f"fx_neu_{league}")
    b = league_bundle(league, years, half_life, wx_user, fuse_on)
    if b is None:
        st.error("Could not load data for this league (check the internet connection of the host).")
    else:
        teams = b["teams"]
        c = st.columns(2)
        h = c[0].selectbox("Home team", teams, index=0, key=f"fx_h_{league}")
        a = c[1].selectbox("Away team", teams, index=min(1, len(teams) - 1), key=f"fx_a_{league}")
        c = st.columns(3)
        o1 = c[0].number_input("Home odds (optional)", 0.0, 100.0, 0.0, 0.01, key="fx_o1")
        ox = c[1].number_input("Draw odds (optional)", 0.0, 100.0, 0.0, 0.01, key="fx_ox")
        o2 = c[2].number_input("Away odds (optional)", 0.0, 100.0, 0.0, 0.01, key="fx_o2")
        if st.button("Add this match", key="fx_add"):
            if h == a:
                st.warning("Choose two different teams.")
            elif db_add_fixture(league, h, a, kick, neutral, o1, ox, o2):
                st.success(f"Added {h} vs {a}.")
            else:
                st.info("This match is already in the list.")

    with st.expander("⚡ Paste a list of matches (fastest way)"):
        st.caption("One match per line, like `Arsenal - Chelsea` or `Premier League: Arsenal vs Chelsea`. "
                   "Team names are matched automatically, spelling does not have to be exact.")
        txt = st.text_area("Matches", height=140, key="fx_paste")
        d2 = st.date_input("Date for these matches", value=dt.date.today(), key="fx_paste_date")
        if st.button("Add pasted matches", key="fx_paste_btn"):
            added, bad = 0, []
            for line in [x.strip() for x in txt.splitlines() if x.strip()]:
                lg, body = league, line
                if ":" in line:
                    pre, rest = line.split(":", 1)
                    guess = match_team(pre.strip(), BOARD_LEAGUES)
                    if guess:
                        lg, body = guess, rest
                parts = re.split(r"\s+(?:vs\.?|v|-|–|—)\s+", body.strip(), maxsplit=1, flags=re.I)
                bb = league_bundle(lg, years, half_life, wx_user, fuse_on)
                if len(parts) != 2 or bb is None:
                    bad.append(line)
                    continue
                hh, aa = match_team(parts[0].strip(), bb["teams"]), match_team(parts[1].strip(), bb["teams"])
                if not hh or not aa or hh == aa:
                    bad.append(line)
                    continue
                added += db_add_fixture(lg, hh, aa, d2, lg.endswith("(national teams)"))
            st.success(f"Added {added} match(es).")
            if bad:
                st.warning("Could not understand: " + " | ".join(bad))

    with st.expander("📡 Import upcoming matches automatically"):
        st.caption("Pulls the published fixtures (with market odds when available) for the main leagues. "
                   "Leagues without a free fixture feed need manual entry above.")
        days = st.slider("Matches in the next N days", 1, 14, 5, key="imp_days")
        fd_leagues = [k for k in BOARD_LEAGUES if SOURCES[k][0] == "fd"]
        pick = st.multiselect("Leagues", fd_leagues, default=[], key="imp_leagues")
        if st.button("Fetch and add", key="imp_btn") and pick:
            fxd = load_fd_fixtures()
            today = pd.Timestamp.today().normalize()
            added = bad = 0
            for lg in pick:
                bb = league_bundle(lg, years, half_life, wx_user, fuse_on)
                if bb is None:
                    continue
                src_rows = pd.DataFrame()
                if len(fxd):
                    src_rows = fxd[(fxd.League == lg) & (fxd.Date >= today) & (fxd.Date <= today + pd.Timedelta(days=days))]
                if src_rows.empty and len(bb["fut"]):
                    f2 = bb["fut"]
                    src_rows = f2[(f2.Date <= today + pd.Timedelta(days=days))].rename(columns={"HomeTeam": "HomeTeam"})
                    for c_ in ("o1", "ox", "o2"):
                        src_rows[c_] = np.nan
                for r in src_rows.itertuples():
                    hh, aa = match_team(r.HomeTeam, bb["teams"]), match_team(r.AwayTeam, bb["teams"])
                    if not hh or not aa or hh == aa:
                        bad += 1
                        continue
                    added += db_add_fixture(lg, hh, aa, r.Date.date(), 0, r.o1, r.ox, r.o2)
            if added == 0 and bad == 0:
                st.info("No upcoming matches were published for these leagues in that window. Add them manually.")
            else:
                st.success(f"Added {added} match(es)." + (f" {bad} skipped (team not recognised)." if bad else ""))

    st.markdown("#### Match list (database)")
    fxdf = db_read("SELECT id, league, kickoff, home, away, neutral, o1, ox, o2, adj_h, adj_a, note "
                   "FROM fixtures ORDER BY kickoff, league, id")
    if fxdf.empty:
        st.info("The list is empty. Add matches above.")
    else:
        st.dataframe(fxdf, hide_index=True, use_container_width=True)
        c = st.columns([3, 1, 1])
        ids = c[0].multiselect("Remove matches (by id)", fxdf["id"].tolist(), key="fx_del")
        if c[1].button("Remove selected", key="fx_del_btn") and ids:
            for i in ids:
                db_exec("DELETE FROM fixtures WHERE id=?", (int(i),))
                db_exec("DELETE FROM picks WHERE fixture_id=?", (int(i),))
            st.success("Removed. Press R to refresh the list.")
        if c[2].button("Remove ALL", key="fx_del_all"):
            db_exec("DELETE FROM fixtures")
            db_exec("DELETE FROM picks")
            st.success("List cleared.")
    st.caption("The database file lives next to the app. On Streamlit Cloud it is wiped when the app restarts, so "
               "use Backup / Restore in the Data tab.")


def ui_analysis(years, half_life, wx_user, fuse_on):
    fxdf = db_read("SELECT * FROM fixtures ORDER BY kickoff, league, id")
    fxdf = fxdf[(fxdf.kickoff.isna()) | (fxdf.kickoff >= (dt.date.today() - dt.timedelta(days=1)).isoformat())]
    if fxdf.empty:
        st.info("Add upcoming matches in the Matches tab. The analysis then runs automatically.")
        return
    with st.expander("Filters", expanded=True):
        c = st.columns(3)
        allowed = c[0].multiselect("Markets to consider", SEL, default=DEFAULT_MARKETS, key="an_mk")
        min_p = c[1].slider("Minimum probability %", 40, 90, 60, 1, key="an_p") / 100
        min_fair = c[2].slider("Minimum fair odds (skips trivial picks)", 1.05, 2.5, 1.25, 0.05, key="an_o")
    fxs, skipped, bundles = [], [], {}
    for r in fxdf.itertuples():
        if r.league not in bundles:
            bundles[r.league] = league_bundle(r.league, years, half_life, wx_user, fuse_on)
        b = bundles[r.league]
        f = analyse_fixture(r, b) if b else None
        (fxs if f else skipped).append(f if f else f"{r.home} vs {r.away} ({r.league})")
    if skipped:
        st.warning("Could not analyse: " + " | ".join(skipped))
    if not fxs:
        return
    save_picks(fxs, allowed, min_p, min_fair)

    rows = []
    for f in fxs:
        P = {s["sel"]: s["p"] for s in f["sels"]}
        bp = best_pick(f, allowed, min_p, min_fair)
        rows.append({"League": f["league"], "Date": f["kickoff"], "Match": f"{f['home']} vs {f['away']}",
                     "Goals exp.": f"{f['lam']:.2f}-{f['mu']:.2f}", "Likely score": f"{f['tops'][0][0]}-{f['tops'][0][1]}",
                     "1 %": round(P["1"] * 100), "X %": round(P["X"] * 100), "2 %": round(P["2"] * 100),
                     "Over 2.5 %": round(P["Over 2.5"] * 100), "BTTS %": round(P["BTTS Yes"] * 100),
                     "Best pick": bp["sel"] if bp else "-", "Pick %": round(bp["p"] * 100) if bp else None,
                     "Data": f["grade"]})
    st.markdown("#### Predictions for all matches")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Data grade: A = both teams have 30+ recent matches, B = 18+, C = less (be careful). "
               "Pre-match picks are saved so the Track record tab can grade them after the games.")

    legs = top_legs(fxs, allowed, min_p, min_fair)
    st.markdown("#### Best picks (ranked by model probability)")
    if not legs:
        st.warning("No selection passes the filters. Lower the minimum probability or fair odds.")
    else:
        st.dataframe(pd.DataFrame([{"#": i + 1, "League": l["league"], "Match": l["match"], "Pick": l["sel"],
                                    "Model %": round(l["p"] * 100, 1), "Fair odds": round(l["fair"], 2),
                                    "Market odds": l["odds"], "Value %": None if l["ev"] is None else round(l["ev"] * 100, 1),
                                    "Data": fxs[l["fi"]]["grade"]} for i, l in enumerate(legs)]),
                     hide_index=True, use_container_width=True)
        st.markdown("#### Combined slip analysis")
        names = [l["match"] for l in legs]
        chosen_names = st.multiselect("Matches in the slip (default: the strongest 4)", names,
                                      default=names[:4], key="an_slip")
        chosen = [l for l in legs if l["match"] in chosen_names]
        if chosen:
            sizes = []
            for k in range(1, len(chosen) + 1):
                s = slip_stats(chosen[:k])
                sizes.append({"Legs": k, "Chance all win %": round(s["P"] * 100, 1),
                              "Chance slip fails %": round((1 - s["P"]) * 100, 1), "Combined fair odds": round(s["fair"], 2)})
            st.dataframe(pd.DataFrame(sizes), hide_index=True, use_container_width=True)
            s = slip_stats(chosen)
            dist = simulate(fxs, chosen)
            c = st.columns(3)
            c[0].metric("Chance every pick wins", f"{s['P'] * 100:.1f}%")
            c[1].metric("Exactly one fails", f"{(dist[1] * 100 if len(dist) > 1 else 0):.1f}%")
            c[2].metric("Two or more fail", f"{dist[2:].sum() * 100:.1f}%")
            if s["P"] < 0.5:
                st.warning("More likely to fail than to win. Every extra match lowers the chance, so a long slip is "
                           "risky however good each pick looks.")
            st.caption(f"Monte Carlo with {N_SIMS:,} simulated rounds. Matches are treated as independent. "
                       "No outcome is guaranteed.")

    st.markdown("#### Match detail")
    labels = [f"{f['league']} | {f['home']} vs {f['away']} | {f['kickoff']}" for f in fxs]
    sel = st.selectbox("Choose a match", labels, key="an_detail")
    f = fxs[labels.index(sel)]
    st.info(explain(f))
    c = st.columns(3)
    c[0].metric(f"{f['home']} expected goals", f"{f['lam']:.2f}")
    c[1].metric(f"{f['away']} expected goals", f"{f['mu']:.2f}")
    c[2].metric("Data grade", f["grade"], f"{f['nh']} / {f['na']} recent matches", delta_color="off")
    sub = st.tabs(["🎯 Correct score", "🛡️ Double chance", "⚖️ Handicap", "⏱️ Halves", "⚽ Goals & BTTS"])
    with sub[0]:
        render_correct_score(f["m"], f["home"], f["away"])
    with sub[1]:
        render_double_chance(f["m"], f["home"], f["away"])
    with sub[2]:
        render_handicap(f["m"], f["lam"], f["mu"], f["home"], f["away"])
    with sub[3]:
        render_halves(f["lam"], f["mu"], f["home"], f["away"], f["b"]["s_share"], f["b"]["s_src"])
    with sub[4]:
        render_goals(f["m"], f["lam"], f["mu"], f["home"], f["away"])


def ui_assistant(years, half_life, wx_user, fuse_on):
    st.markdown("#### 🤖 AI research assistant")
    st.caption("The statistical model knows results, not news. This assistant searches the web for injuries, "
               "line-ups and form, then you decide whether to adjust the model. It needs an Anthropic API key "
               "(paid per use, usually a few cents per match). Without a key everything else still works.")
    key = ""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = st.text_input("Anthropic API key (kept only in this session)", type="password", key="ai_key")
    c = st.columns(2)
    model_id = c[0].text_input("Model", "claude-sonnet-5-5", key="ai_model")
    max_uses = c[1].slider("Max web searches per match", 1, 8, 4, key="ai_uses")
    fxdf = db_read("SELECT * FROM fixtures ORDER BY kickoff, league, id")
    if fxdf.empty:
        st.info("Add matches first.")
        return
    labels = [f"#{r.id} {r.league} | {r.home} vs {r.away} | {r.kickoff}" for r in fxdf.itertuples()]
    sel = st.selectbox("Match", labels, key="ai_match")
    r = list(fxdf.itertuples())[labels.index(sel)]
    f = {"home": r.home, "away": r.away, "league": r.league, "kickoff": r.kickoff}
    store = st.session_state.setdefault("ai_reports", {})
    if st.button("Search the web for this match", key="ai_go"):
        if not key:
            st.error("Enter an API key first.")
        else:
            with st.spinner("Searching..."):
                txt, cites, err = ai_research(key, model_id, f, max_uses)
            if err:
                st.error(err)
            else:
                store[int(r.id)] = (txt, cites)
    if int(r.id) in store:
        txt, cites = store[int(r.id)]
        st.markdown(txt)
        if cites:
            st.markdown("**Sources:** " + " | ".join(f"[{t}]({u})" for u, t in cites.items()))
        st.caption("This text comes from web pages and a language model. Check it before relying on it.")
    st.markdown("**Adjust the model for this match** (your judgement after reading the report)")
    st.caption("Multiplies a team's expected goals. Example: main striker out about 0.92; weakened rotation 0.9.")
    c = st.columns(3)
    ah = c[0].slider(f"{r.home} attack x", 0.70, 1.20, float(r.adj_h or 1.0), 0.01, key=f"adj_h_{r.id}")
    aa = c[1].slider(f"{r.away} attack x", 0.70, 1.20, float(r.adj_a or 1.0), 0.01, key=f"adj_a_{r.id}")
    note = c[2].text_input("Note", r.note or "", key=f"adj_n_{r.id}")
    if st.button("Save adjustment", key=f"adj_save_{r.id}"):
        db_exec("UPDATE fixtures SET adj_h=?, adj_a=?, note=? WHERE id=?", (float(ah), float(aa), note, int(r.id)))
        st.success("Saved. The Analysis tab now uses it.")


def ui_track(years, half_life, wx_user, fuse_on):
    st.markdown("#### Track record")
    pk = db_read("SELECT p.fixture_id, p.sel, p.p, f.league, f.home, f.away, f.kickoff FROM picks p "
                 "JOIN fixtures f ON f.id = p.fixture_id")
    if pk.empty:
        st.info("Nothing yet. Picks are saved automatically when you open the Analysis tab for upcoming matches.")
        return
    rows = []
    for r in pk.itertuples():
        b = league_bundle(r.league, years, half_life, wx_user, fuse_on)
        res = None
        if b is not None and r.kickoff:
            kd = pd.Timestamp(r.kickoff)
            d = b["df"]
            m = d[(d.HomeTeam == r.home) & (d.AwayTeam == r.away) &
                  (d.Date >= kd - pd.Timedelta(days=2)) & (d.Date <= kd + pd.Timedelta(days=3))]
            if len(m):
                res = (int(m.iloc[0].FTHG), int(m.iloc[0].FTAG))
        rows.append({"Date": r.kickoff, "Match": f"{r.home} vs {r.away}", "Pick": r.sel, "Model %": round(r.p * 100, 1),
                     "Result": "pending" if res is None else f"{res[0]}-{res[1]}",
                     "Hit": None if res is None else sel_hit(r.sel, *res), "_p": r.p})
    t = pd.DataFrame(rows).sort_values("Date", ascending=False)
    g = t[t.Hit.notna()].copy()
    if g.empty:
        st.info("Matches are still pending. Results appear automatically once the data source publishes them "
                "(usually within a day).")
    else:
        g["Hit"] = g["Hit"].astype(bool)
        c = st.columns(3)
        c[0].metric("Graded picks", len(g))
        c[1].metric("Hit rate", f"{g.Hit.mean() * 100:.1f}%")
        c[2].metric("Average model probability", f"{g._p.mean() * 100:.1f}%")
        g["Bucket"] = pd.cut(g._p, [0, .5, .6, .7, .8, 1], labels=["<50", "50-60", "60-70", "70-80", "80+"])
        cal = g.groupby("Bucket", observed=True).agg(Picks=("Hit", "size"), Hit_rate=("Hit", "mean"), Model=("_p", "mean"))
        cal["Hit_rate"] = (cal["Hit_rate"] * 100).round(1)
        cal["Model"] = (cal["Model"] * 100).round(1)
        st.markdown("**Calibration** (hit rate should be close to the model percentage; small samples are noisy)")
        st.dataframe(cal.rename(columns={"Hit_rate": "Hit rate %", "Model": "Model %"}), use_container_width=True)
    st.dataframe(t.drop(columns="_p"), hide_index=True, use_container_width=True)


def ui_data(years, half_life, wx_user, fuse_on):
    st.markdown("#### Data sources and backup")
    leagues = st.multiselect("Show status for leagues", BOARD_LEAGUES, default=[], key="dt_leagues")
    rows = []
    for lg in leagues:
        b = league_bundle(lg, years, half_life, wx_user, fuse_on)
        if b is None:
            rows.append({"League": lg, "Status": "no data"})
            continue
        d = b["df"]
        rows.append({"League": lg, "Matches": len(d), "From": d.Date.min().date(), "To": d.Date.max().date(),
                     "Teams": len(b["teams"]), "xG source": b["xg"], "Added by fusion": b["added"],
                     "Upcoming (2nd source)": len(b["fut"])})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Sources: football-data.co.uk (results, shots, odds), openfootball on GitHub (second results feed, "
               "fused by team-name matching), martj42 international results (national teams).")
    fx = db_read("SELECT * FROM fixtures ORDER BY id")
    st.download_button("⬇️ Backup match list (CSV)", fx.to_csv(index=False), "board_backup.csv", "text/csv", key="bk_dl")
    up = st.file_uploader("Restore match list from a backup CSV", type="csv", key="bk_up")
    if up is not None and st.button("Restore", key="bk_btn"):
        try:
            d = pd.read_csv(up)
            n = 0
            for r in d.itertuples():
                n += db_add_fixture(r.league, r.home, r.away, r.kickoff, getattr(r, "neutral", 0),
                                    getattr(r, "o1", None), getattr(r, "ox", None), getattr(r, "o2", None))
            st.success(f"Restored {n} match(es).")
        except Exception as e:
            st.error(f"Could not read the backup: {e}")


def render_board(years, half_life, wx_user, fuse_on):
    st.markdown("### 🗂️ Match board: many leagues, one analysis")
    st.caption("Add matches from any league, the engine trains each league's model, predicts every match, ranks the "
               "strongest picks and shows how risky a combined slip really is. Analysis only, no betting on this site.")
    try:
        _con().close()
    except sqlite3.Error as e:
        st.error(f"Database problem: {e}")
        return
    tabs = st.tabs(["➕ Matches", "📊 Analysis", "🤖 AI assistant", "📈 Track record", "🧰 Data & backup"])
    with tabs[0]:
        ui_add_fixtures(years, half_life, wx_user, fuse_on)
    with tabs[1]:
        ui_analysis(years, half_life, wx_user, fuse_on)
    with tabs[2]:
        ui_assistant(years, half_life, wx_user, fuse_on)
    with tabs[3]:
        ui_track(years, half_life, wx_user, fuse_on)
    with tabs[4]:
        ui_data(years, half_life, wx_user, fuse_on)


# ======================= UI =======================
st.set_page_config(page_title="Pro Football AI Engine", layout="wide")
st.title("⚽ Pro Football AI Engine")
st.caption("Time-weighted Dixon-Coles model trained on real results, shots and xG. Analysis only, no betting.")

sb = st.sidebar
page = sb.radio("Page", ["🗂️ Match board (multi-league)", "🎯 Single match deep-dive", "👥 Players"], key="page")
sb.header("Settings")
years = sb.slider("Years of history", 1, 6, 3)
half_life = sb.slider("Recency half-life (days)", 60, 900, 300, 30)
wx_user = sb.slider("xG / shots weight (0 = goals only)", 0.0, 1.0, 0.5, 0.05)
fuse_on = sb.checkbox("Fuse a second data source (openfootball)", value=True)

if page.startswith("🗂️"):
    render_board(years, half_life, wx_user, fuse_on)
    st.stop()
if page.startswith("👥"):
    render_players_tab()
    st.stop()

src = sb.selectbox("Competition", list(SOURCES))
kind, code = SOURCES[src]

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
    if fuse_on and kind == "fd" and not df.empty:
        _extra, _ = load_openfootball(code, years)
        if not _extra.empty:
            df, _added = fuse(df, [_extra])
            sb.caption(f"Fusion: +{_added} matches from the second source")

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
