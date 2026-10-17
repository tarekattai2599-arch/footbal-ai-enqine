import datetime as dt
import difflib
import io
import json
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import poisson

APP_VERSION = "v11"
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
    "League One (England)": ("fd", "E2"), "League Two (England)": ("fd", "E3"),
    "Scottish Championship": ("fd", "SC1"), "Bundesliga 2 (Germany)": ("fd", "D2"),
    "Serie B (Italy)": ("fd", "I2"), "La Liga 2 (Spain)": ("fd", "SP2"), "Ligue 2 (France)": ("fd", "F2"),
    "China Super League": ("new", "CHN"), "Finland Veikkausliiga": ("new", "FIN"),
    "Ireland Premier Division": ("new", "IRL"), "Romania Liga 1": ("new", "ROU"),
    "Africa Cup of Nations (national teams)": ("intl", "African Cup of Nations"),
    "UEFA Nations League (national teams)": ("intl", "UEFA Nations League"),
    "Champions League (football-data.org free key)": ("fdorg", "CL"),
    "Egyptian Premier League (upload CSV)": ("upload", ""),
    "Saudi Pro League (upload CSV)": ("upload", ""),
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
# ---------- robust downloading: browser-like header, retries, saved copy on disk, error log ----------
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36", "Accept": "*/*"}
CACHE_DIR = "data_cache"
NET_LOG = {}          # key -> (status, detail)
HOST_DEAD = {}        # host -> time of the last connection failure (skip it for a short while)


def fetch_bytes(url, tries=2, timeout=20):
    last = ""
    host = url.split("/")[2] if "//" in url else url
    if time.time() - HOST_DEAD.get(host, 0) < 60:
        return None, f"{host} did not answer a moment ago (skipped, press 'Try again now')"
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 100:
                return r.content, ""
            last = f"HTTP {r.status_code}"
            if r.status_code in (403, 404):
                break
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:110]}"
            if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                HOST_DEAD[host] = time.time()
                break
        time.sleep(0.8 * (i + 1))
    return None, last


def fetch_csv(url, key):
    """Download a CSV. On success keep a copy on disk; on failure fall back to the last saved copy."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    path = os.path.join(CACHE_DIR, safe + ".csv")
    raw, err = fetch_bytes(url)
    if raw is not None:
        NET_LOG[key] = ("ok", url)
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "wb") as f:
                f.write(raw)
        except OSError:
            pass
    elif os.path.exists(path):
        try:
            with open(path, "rb") as f:
                raw = f.read()
            NET_LOG[key] = ("saved copy", f"{err} - using the last saved copy")
        except OSError:
            raw = None
    if raw is None:
        NET_LOG[key] = ("failed", f"{err}  [{url}]")
        return None
    try:
        try:
            txt = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            txt = raw.decode("latin-1")
        return pd.read_csv(io.StringIO(txt), on_bad_lines="skip")
    except Exception as e:
        NET_LOG[key] = ("failed", f"unreadable file: {str(e)[:100]}  [{url}]")
        return None


def net_failures(prefix):
    return [f"{k}: {v[1]}" for k, v in NET_LOG.items() if k.startswith(prefix) and v[0] == "failed"]

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
def _load_data_raw(kind, code, years, key=""):
    """Raises when nothing could be loaded, so a failure is never cached."""
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=int(365.25 * years))
    if kind == "fd":
        y0 = today.year if today.month >= 7 else today.year - 1
        frames = []
        for y in range(y0, y0 - years - 1, -1):
            tag = f"{y % 100:02d}{(y + 1) % 100:02d}"
            d = fetch_csv(f"{FD}/mmz4281/{tag}/{code}.csv", f"{code}-{tag}")
            if d is not None and len(d):
                frames.append(d)
        if not frames:
            raise RuntimeError("football-data.co.uk: no season file could be downloaded")
        df = clean(normalize(pd.concat(frames, ignore_index=True)))
    elif kind == "new":
        d = fetch_csv(f"{FD}/new/{code}.csv", f"new-{code}")
        if d is None:
            raise RuntimeError("football-data.co.uk: file could not be downloaded")
        df = clean(normalize(d))
    elif kind == "fdorg":
        df = load_fdorg(code, years, key)[0]
    else:
        d = fetch_csv(INTL_URL, "intl-results")
        if d is None:
            raise RuntimeError("international results file could not be downloaded")
        df = clean(normalize(d), dayfirst=False)
    df = df[df.Date >= cutoff].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("the file was downloaded but has no recent matches")
    return df


BACKUP_USED = {}


def load_data(kind, code, years, key=""):
    """Main source first; for the big European leagues fall back to the openfootball source if it fails."""
    try:
        return _load_data_raw(kind, code, years, key)
    except Exception:
        pass
    if kind == "fd" and code in OF:
        res, _ = load_openfootball(code, years)
        if not res.empty:
            BACKUP_USED[code] = len(res)
            return res
    return pd.DataFrame()


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


# ======================= ELO (second, independent model) =======================
def elo_to_1x2(E, d0):
    """Win expectancy -> (home, draw, away). The draw share is largest between equal teams."""
    d = float(np.clip(d0 * 4 * E * (1 - E), 0.03, 0.45))
    ph, pa = max(E - d / 2, 0.01), max(1 - E - d / 2, 0.01)
    t = ph + d + pa
    return ph / t, d / t, pa / t


def _elo_pass(df, K, hfa, record_from=None):
    r, out = {}, []
    neu = df["Neutral"].values if "Neutral" in df else np.zeros(len(df))
    for i, (h, a, hg, ag) in enumerate(zip(df.HomeTeam.values, df.AwayTeam.values, df.FTHG.values, df.FTAG.values)):
        rh, ra = r.get(h, 1500.0), r.get(a, 1500.0)
        E = 1 / (1 + 10 ** (-(rh - ra + (0 if neu[i] else hfa)) / 400))
        y = 0 if hg > ag else (1 if hg == ag else 2)
        if record_from is not None and i >= record_from:
            out.append((h, a, E, y))
        gd = abs(hg - ag)
        G = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        delta = K * G * ((1.0 if y == 0 else 0.5 if y == 1 else 0.0) - E)
        r[h], r[a] = rh + delta, ra - delta
    return r, out


def fit_elo(df):
    """Pick the Elo speed (K) and home bonus that best predicted this league's own history."""
    d0 = float((df.FTHG == df.FTAG).mean()) if len(df) else 0.26
    if len(df) < 80:
        return {"K": 20, "hfa": 80, "d0": d0, "ratings": _elo_pass(df, 20, 80)[0]}
    best = None
    for K in (10, 20, 35):
        for hfa in (50, 80, 110):
            _, out = _elo_pass(df, K, hfa, record_from=int(0.3 * len(df)))
            ll = -np.mean([np.log(max(elo_to_1x2(E, d0)[y], 1e-9)) for _, _, E, y in out])
            if best is None or ll < best[0]:
                best = (ll, K, hfa)
    return {"K": best[1], "hfa": best[2], "d0": d0, "ratings": _elo_pass(df, best[1], best[2])[0]}


def elo_probs(elo, h, a, neutral=False):
    if not elo:
        return None
    r = elo["ratings"]
    E = 1 / (1 + 10 ** (-(r.get(h, 1500.0) - r.get(a, 1500.0) + (0 if neutral else elo["hfa"])) / 400))
    return elo_to_1x2(E, elo["d0"])


def elo_walk(df, n_test):
    """Walk-forward Elo forecasts for the last n_test matches (ratings update only after each match)."""
    cut = len(df) - n_test
    cfg = fit_elo(df.iloc[:cut])
    _, out = _elo_pass(df, cfg["K"], cfg["hfa"], record_from=cut)
    return {cut + i: elo_to_1x2(E, cfg["d0"]) for i, (_, _, E, _) in enumerate(out)}


def blend_with_market(p_model, mk, w):
    """w = weight on the model, (1 - w) on the de-vigged market probabilities."""
    p = w * np.asarray(p_model, dtype=float) + (1 - w) * np.asarray(mk, dtype=float)
    return p / p.sum(axis=-1, keepdims=True)


def tilt_matrix(m, target):
    """Rescale a score matrix so its 1X2 probabilities equal `target`, keeping the score shape inside each outcome."""
    n = m.shape[0]
    i, j = np.indices((n, n))
    out = m.copy()
    for mask, t in ((i > j, target[0]), (i == j, target[1]), (i < j, target[2])):
        tot = m[mask].sum()
        if tot > 0:
            out[mask] = m[mask] * t / tot
    return out / out.sum()


def wf_logloss(df, xi, wx, n_test=150, chunks=2):
    """Walk-forward log loss of the 1X2 forecast on the most recent matches (lower = better)."""
    n_test = min(n_test, len(df) // 3)
    ll = []
    for s in np.array_split(np.arange(len(df) - n_test, len(df)), chunks):
        md = fit_dc(df.iloc[:s[0]], xi, wx)
        for k in s:
            r = df.iloc[k]
            if r.HomeTeam not in md["idx"] or r.AwayTeam not in md["idx"]:
                continue
            lam, mu = lambdas(md, r.HomeTeam, r.AwayTeam, bool(r.get("Neutral", 0)))
            p = outcome_probs(score_matrix(lam, mu, md["rho"]))
            y = 0 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 2)
            ll.append(-np.log(max(p[y], 1e-9)))
    return (float(np.mean(ll)) if ll else float("nan")), len(ll)


def backtest(df, xi, n_test, wx, chunks=3):
    n_test = min(n_test, len(df) // 3)
    names = {"main": "Our model + xG/shots" if wx > 0 else "Our model (goals only)"}
    wxs = {"main": wx}
    if wx > 0:
        names["base"], wxs["base"] = "Our model (goals only)", 0.0
    P, Y, MK, EL = {k: [] for k in names}, [], [], []
    elo_tbl = elo_walk(df, n_test)
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
            EL.append(elo_tbl[k])
    if not Y:
        return None
    tr = df.iloc[:len(df) - n_test]
    freq = [(tr.FTHG > tr.FTAG).mean(), (tr.FTHG == tr.FTAG).mean(), (tr.FTHG < tr.FTAG).mean()]
    out = {names[k]: metrics(P[k], Y) for k in names}
    out["Naive (league averages)"] = metrics([freq] * len(Y), Y)
    El, Pm_all = np.array(EL), np.array(P["main"])
    out["Elo (own)"] = metrics(El, Y)
    out["Model + Elo 20%"] = metrics(blend_with_market(Pm_all, El, 0.8), Y)
    bw = min(np.arange(0, 1.01, 0.1), key=lambda w: metrics(blend_with_market(Pm_all, El, w), Y)["Brier (lower=better)"])
    out[f"Model + Elo best (model {bw * 100:.0f}%)"] = metrics(blend_with_market(Pm_all, El, bw), Y)
    sub = [i for i, m in enumerate(MK) if m is not None]
    if sub:
        out["Our model (matches with odds)"] = metrics([P["main"][i] for i in sub], [Y[i] for i in sub])
        out["Bookmaker odds"] = metrics([MK[i] for i in sub], [Y[i] for i in sub])
        Pm, Pk, Ys = np.array([P["main"][i] for i in sub]), np.array([MK[i] for i in sub]), [Y[i] for i in sub]
        out["Blend 50/50 (model + market)"] = metrics(blend_with_market(Pm, Pk, 0.5), Ys)
        best = min(np.arange(0, 1.01, 0.1), key=lambda w: metrics(blend_with_market(Pm, Pk, w), Ys)["Brier (lower=better)"])
        out[f"Blend best weight (model {best * 100:.0f}%)"] = metrics(blend_with_market(Pm, Pk, best), Ys)
    return out


# ======================= CALIBRATION: make the confidence honest =======================
CAL_DEF = {"w_elo": 0.20, "temp": 1.0, "pa": 0.0, "pb": 1.0, "ba": 0.0, "bb": 1.0}


def _temp(p, T):
    """Temperature: T > 1 makes forecasts less extreme, T < 1 more extreme."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, None) ** (1.0 / T)
    return p / p.sum(axis=-1, keepdims=True)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _platt(p, a, b):
    return 1 / (1 + np.exp(-(a + b * _logit(np.asarray(p, dtype=float)))))


def _ll3(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, None))))


def _ll2(p, t):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def wf_collect(df, xi, wx, n_test=300, chunks=3):
    """Walk-forward forecasts (model refitted on the past only) for the most recent matches of a league."""
    n_test = min(n_test, len(df) // 3)
    elo_tbl = elo_walk(df, n_test)
    o = {k: [] for k in ("dc", "elo", "y", "over", "ov", "btts", "bt")}
    for s_ in np.array_split(np.arange(len(df) - n_test, len(df)), chunks):
        md = fit_dc(df.iloc[:s_[0]], xi, wx)
        for k in s_:
            r = df.iloc[k]
            if r.HomeTeam not in md["idx"] or r.AwayTeam not in md["idx"]:
                continue
            lam, mu = lambdas(md, r.HomeTeam, r.AwayTeam, bool(r.get("Neutral", 0)))
            m = score_matrix(lam, mu, md["rho"])
            ii, jj = np.indices(m.shape)
            o["dc"].append(outcome_probs(m))
            o["elo"].append(elo_tbl[k])
            o["y"].append(0 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 2))
            o["over"].append(float(m[(ii + jj) > 2].sum()))
            o["ov"].append(int(r.FTHG + r.FTAG > 2))
            o["btts"].append(float(m[(ii > 0) & (jj > 0)].sum()))
            o["bt"].append(int(r.FTHG > 0 and r.FTAG > 0))
    return {k: np.array(v) for k, v in o.items()}


def fit_calibration(W):
    """Pick the Elo weight and a confidence correction that would have predicted this league's latest matches best.
    A setting only replaces the default when the gain is statistically clear (total log-likelihood gain of 4+,
    roughly a 1-in-40 chance of being noise), so the app does not chase luck.
    Returns (settings, reliability info or None)."""
    y = W["y"]
    n = len(y)
    res = {**CAL_DEF, "n": int(n)}
    if n < 120:
        return res, None

    def final(we, T):
        return _temp((1 - we) * W["dc"] + we * W["elo"], T)
    base = _ll3(final(0.2, 1.0), y)
    best = (base, 0.2, 1.0)
    for we in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        for T in (0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2):
            ll = _ll3(final(we, T), y)
            if ll < best[0] - 1e-12:
                best = (ll, we, T)
    if (base - best[0]) * n >= 4.0:
        res["w_elo"], res["temp"] = best[1], best[2]
        use_ll = best[0]
    else:
        use_ll = base
    res["loss"], res["base_loss"] = use_ll, base
    for key, pk, yk in (("p", "over", "ov"), ("b", "btts", "bt")):
        p, t = W[pk], W[yk]
        b0 = _ll2(p, t)
        bb = (b0, 0.0, 1.0)
        for a in np.arange(-0.4, 0.41, 0.1):
            for bsl in (0.8, 0.9, 1.0, 1.1, 1.2):
                l2 = _ll2(_platt(p, a, bsl), t)
                if l2 < bb[0] - 1e-12:
                    bb = (l2, float(a), float(bsl))
        if (b0 - bb[0]) * len(t) >= 4.0:
            res[key + "a"], res[key + "b"] = bb[1], bb[2]
    P = final(res["w_elo"], res["temp"])
    top, hit = P.max(1), (P.argmax(1) == y)
    rel = []
    edges = [0, 0.40, 0.50, 0.60, 0.70, 1.01]
    for lo, hi, lb in zip(edges[:-1], edges[1:], ["under 40%", "40-50%", "50-60%", "60-70%", "70%+"]):
        sel = (top >= lo) & (top < hi)
        if sel.sum():
            rel.append({"Model confidence": lb, "Matches": int(sel.sum()),
                        "Model said %": round(float(top[sel].mean()) * 100, 1),
                        "Really happened %": round(float(hit[sel].mean()) * 100, 1)})
    k = max(1, n // 3)
    order = np.argsort(-top)
    info = {"rel": rel, "all_acc": round(float(hit.mean()) * 100, 1), "n": int(n),
            "top_acc": round(float(hit[order[:k]].mean()) * 100, 1), "top_n": int(k),
            "top_min": round(float(top[order[k - 1]]) * 100, 1)}
    return res, info


def eff_params(src, half_life, wx_user):
    try:
        p = db_read("SELECT half_life, wx FROM league_params WHERE league=?", (src,))
        if len(p):
            return int(p.half_life[0]), float(p.wx[0])
    except Exception:
        pass
    return half_life, wx_user


def run_calibration(lg, b, half_life, wx_user):
    has_xg = not b["xg"].startswith("not available")
    W = wf_collect(b["df"], float(np.log(2) / half_life), wx_user if has_xg else 0.0)
    res, info = fit_calibration(W)
    db_exec("INSERT OR REPLACE INTO league_calib(league,w_elo,temp,pa,pb,ba,bb,loss,base_loss,n,rel,tuned_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (lg, res["w_elo"], res["temp"], res["pa"], res["pb"], res["ba"], res["bb"], res.get("loss"),
             res.get("base_loss"), res["n"], json.dumps(info) if info else None, dt.datetime.now().isoformat()))
    return res, info


def calib_load(league):
    try:
        d = db_read("SELECT * FROM league_calib WHERE league=?", (league,))
    except Exception:
        return None
    if not len(d):
        return None
    r = d.iloc[0]
    out = {k: float(r[k]) for k in CAL_DEF}
    out["n"] = int(r["n"]) if pd.notna(r["n"]) else 0
    out["loss"], out["base_loss"] = r["loss"], r["base_loss"]
    try:
        out["info"] = json.loads(r["rel"]) if r["rel"] else None
    except Exception:
        out["info"] = None
    return out


def _cal_sel(k, p, c):
    """Apply the saved correction to goal-market probabilities."""
    if k == "Over 2.5":
        return float(_platt(p, c["pa"], c["pb"]))
    if k == "Under 2.5":
        return 1 - float(_platt(1 - p, c["pa"], c["pb"]))
    if k == "BTTS Yes":
        return float(_platt(p, c["ba"], c["bb"]))
    if k == "BTTS No":
        return 1 - float(_platt(1 - p, c["ba"], c["bb"]))
    return p


def render_reliability(lg, calib):
    info = calib.get("info") if calib else None
    if not info:
        st.caption("This league is not calibrated yet.")
        return
    c = st.columns(3)
    c[0].metric("Result correct, all matches", f"{info['all_acc']}%", f"last {info['n']} matches", delta_color="off")
    c[1].metric("Result correct, most confident third", f"{info['top_acc']}%",
                f"model confidence {info['top_min']}%+", delta_color="off")
    c[2].metric("Settings", f"Elo {calib['w_elo'] * 100:.0f}%", f"temperature {calib['temp']:.2f}", delta_color="off")
    st.dataframe(pd.DataFrame(info["rel"]), hide_index=True, use_container_width=True)
    st.caption("Measured on this league's most recent matches, each forecast made without knowing the result. "
               "If 'Model said' and 'Really happened' are close, the percentages can be trusted. Rows with few matches are noisy.")


def agreement(f, pick):
    """Do all result sources (Dixon-Coles, Elo, ClubElo, market) point the same way as the pick?"""
    srcs = list(f["sources"].values())
    sel = pick["sel"]
    if len(srcs) < 2 or sel not in ("1", "X", "2", "1X", "X2", "12"):
        return "n/a"
    if sel in ("1", "X", "2"):
        i = {"1": 0, "X": 1, "2": 2}[sel]
        return "all agree" if all(int(np.argmax(v)) == i for v in srcs) else "split"
    cols = {"1X": (0, 1), "X2": (1, 2), "12": (0, 2)}[sel]
    return "all agree" if all(sum(v[c] for c in cols) >= pick["p"] - 0.08 for v in srcs) else "split"


def verdict(rk, agree, f):
    if rk["risk"] <= 25 and agree != "split" and f["grade"] != "C":
        return "Strong"
    if rk["risk"] <= 40 and agree != "split":
        return "OK"
    return "Skip"


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


PALIAS = {"player": ["player", "Player", "player_name", "name"], "team": ["team", "Team", "team_title"],
          "pos": ["position", "pos", "Pos"], "minutes": ["minutes", "time", "Min"], "goals": ["goals"],
          "xg": ["xg", "xG"], "npxg": ["np_xg", "npxG", "npxg"], "assists": ["assists"], "xa": ["xa", "xA"],
          "shots": ["shots"], "kp": ["key_passes", "kp"], "chain": ["xg_chain", "xGChain"],
          "build": ["xg_buildup", "xGBuildup"]}


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
    st.session_state["players_df"] = d
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
BOARD_LEAGUES = [k for k, (kd, _) in SOURCES.items() if kd in ("fd", "new", "intl", "fdorg")]
FD_CODE_TO_LEAGUE = {code: name for name, (kd, code) in SOURCES.items() if kd == "fd"}
OFB = "https://raw.githubusercontent.com/openfootball/football.json/master"
OF = {"E0": "en.1", "E1": "en.2", "SP1": "es.1", "I1": "it.1", "D1": "de.1", "F1": "fr.1", "N1": "nl.1",
      "P1": "pt.1", "T1": "tr.1", "B1": "be.1", "SC0": "sco.1", "G1": "gr.1",
      "E2": "en.3", "D2": "de.2", "SP2": "es.2", "I2": "it.2", "F2": "fr.2"}
FX_ODDS = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
DEFAULT_MARKETS = ["1", "X", "2", "1X", "X2", "12", "Over 1.5", "Over 2.5", "Under 3.5", "BTTS Yes", "BTTS No"]


# ---------- football-data.org (free key): Champions League results and fixtures ----------
def org_key():
    try:
        k = st.secrets.get("FOOTBALL_DATA_ORG_KEY", "")
    except Exception:
        k = ""
    return k or st.session_state.get("fdorg_key", "")


def parse_fdorg(matches):
    res, fut = [], []
    for m in matches:
        try:
            row = {"Date": str(m["utcDate"])[:10], "HomeTeam": m["homeTeam"]["name"], "AwayTeam": m["awayTeam"]["name"]}
        except (KeyError, TypeError):
            continue
        ft = (m.get("score") or {}).get("fullTime") or {}
        if m.get("status") == "FINISHED" and ft.get("home") is not None and ft.get("away") is not None:
            res.append({**row, "FTHG": ft["home"], "FTAG": ft["away"]})
        elif m.get("status") in ("TIMED", "SCHEDULED"):
            fut.append(row)
    return res, fut


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def load_fdorg(code, years, key):
    """Results (and upcoming fixtures) from api.football-data.org. Needs a free key."""
    if not key:
        return pd.DataFrame(), pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    y0 = today.year if today.month >= 7 else today.year - 1
    res, fut = [], []
    for y in range(y0, y0 - min(years, 3) - 1, -1):
        try:
            r = requests.get(f"https://api.football-data.org/v4/competitions/{code}/matches", params={"season": y},
                             headers={"X-Auth-Token": key}, timeout=25)
            if r.status_code != 200:
                continue
            a, b = parse_fdorg(r.json().get("matches", []))
        except Exception:
            continue
        res += a
        fut += b
    rdf = clean(normalize(pd.DataFrame(res)), dayfirst=False) if res else pd.DataFrame()
    fdf = pd.DataFrame(fut)
    if len(fdf):
        fdf["Date"] = pd.to_datetime(fdf["Date"], errors="coerce")
        fdf = fdf.dropna(subset=["Date"])
        fdf = fdf[fdf.Date >= today]
    return rdf, fdf


# ---------- free external Elo ratings (ClubElo) ----------
CLUBELO_COUNTRY = {"Premier League (England)": "ENG", "Championship (England)": "ENG", "La Liga (Spain)": "ESP",
                   "Serie A (Italy)": "ITA", "Bundesliga (Germany)": "GER", "Ligue 1 (France)": "FRA",
                   "Eredivisie (Netherlands)": "NED", "Liga Portugal": "POR", "Super Lig (Turkey)": "TUR",
                   "Belgian Pro League": "BEL", "Scottish Premiership": "SCO", "Greek Super League": "GRE",
                   "Norway Eliteserien": "NOR", "Sweden Allsvenskan": "SWE", "Russian Premier League": "RUS",
                   "Denmark Superliga": "DEN", "Austria Bundesliga": "AUT", "Switzerland Super League": "SUI",
                   "Poland Ekstraklasa": "POL", "League One (England)": "ENG", "League Two (England)": "ENG",
                   "Scottish Championship": "SCO", "Bundesliga 2 (Germany)": "GER", "Serie B (Italy)": "ITA",
                   "La Liga 2 (Spain)": "ESP", "Ligue 2 (France)": "FRA", "Finland Veikkausliiga": "FIN",
                   "Ireland Premier Division": "IRL", "Romania Liga 1": "ROU"}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_clubelo(day):
    raw, err = fetch_bytes(f"http://api.clubelo.com/{day}", tries=2, timeout=20)
    if raw is None:
        raise RuntimeError(f"ClubElo: {err}")
    d = pd.read_csv(io.StringIO(raw.decode("utf-8", "replace")))
    return d[["Club", "Country", "Elo"]].dropna()


def clubelo_probs(league, home, away, neutral, d0):
    """1X2 from ClubElo ratings (free, cross-league, long history). None if teams are not found."""
    try:
        ce = load_clubelo(dt.date.today().isoformat())
    except Exception:
        return None
    cc = CLUBELO_COUNTRY.get(league)
    pool = ce[ce.Country == cc] if cc else ce
    if pool.empty:
        return None
    names = pool.Club.tolist()
    h, a = match_team(home, names), match_team(away, names)
    if not h or not a or h == a:
        return None
    eh, ea = float(pool[pool.Club == h].Elo.iloc[0]), float(pool[pool.Club == a].Elo.iloc[0])
    return elo_to_1x2(1 / (1 + 10 ** (-(eh - ea + (0 if neutral else 100)) / 400)), d0)


# ---------- database ----------
def _con():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("CREATE TABLE IF NOT EXISTS fixtures(id INTEGER PRIMARY KEY AUTOINCREMENT, league TEXT, home TEXT, "
                "away TEXT, kickoff TEXT, neutral INTEGER DEFAULT 0, o1 REAL, ox REAL, o2 REAL, "
                "adj_h REAL DEFAULT 1, adj_a REAL DEFAULT 1, note TEXT DEFAULT '', created TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS league_params(league TEXT PRIMARY KEY, half_life INTEGER, wx REAL, "
                "loss REAL, base_loss REAL, tuned_at TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS league_calib(league TEXT PRIMARY KEY, w_elo REAL, temp REAL, pa REAL, "
                "pb REAL, ba REAL, bb REAL, loss REAL, base_loss REAL, n INTEGER, rel TEXT, tuned_at TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS context(fixture_id INTEGER PRIMARY KEY, data TEXT, fetched TEXT)")
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


def load_openfootball(code, years):
    try:
        return _load_openfootball_raw(code, years)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_openfootball_raw(code, years):
    """Second free source (openfootball on GitHub): results and, when published, upcoming matches.
    Raises when nothing was downloaded, so a failure is never cached."""
    f = OF.get(code)
    if not f:
        return pd.DataFrame(), pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    y0 = today.year if today.month >= 7 else today.year - 1
    res, fut = [], []
    for y in range(y0, max(y0 - years, y0 - 3) - 1, -1):
        raw, err = fetch_bytes(f"{OFB}/{y}-{(y + 1) % 100:02d}/{f}.json", tries=2)
        if raw is None:
            NET_LOG[f"openfootball-{code}-{y}"] = ("failed", err)
            continue
        try:
            matches = json.loads(raw.decode("utf-8", "replace")).get("matches", [])
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
    if not res and not fut:
        raise RuntimeError("openfootball: nothing downloaded")
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
    d = fetch_csv(f"{FD}/fixtures.csv", "fixtures")
    if d is None:
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


def _season_tag(shift=0):
    t = pd.Timestamp.today()
    y = (t.year if t.month >= 7 else t.year - 1) + shift
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def _season_dash(shift=0):
    t = pd.Timestamp.today()
    y = (t.year if t.month >= 7 else t.year - 1) + shift
    return f"{y}-{(y + 1) % 100:02d}"


def league_bundle(src, years, half_life, wx_user, fuse_on):
    """Uses the league's auto-tuned settings (Data tab) when they exist, otherwise the sidebar values."""
    try:
        p = db_read("SELECT half_life, wx FROM league_params WHERE league=?", (src,))
        if len(p):
            half_life, wx_user = int(p.half_life[0]), float(p.wx[0])
    except Exception:
        pass
    kd = SOURCES[src][0]
    try:
        return _league_bundle(src, years, half_life, wx_user, fuse_on, org_key() if kd == "fdorg" else "")
    except RuntimeError:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner="Loading league and training its model...")
def _league_bundle(src, years, half_life, wx_user, fuse_on, org=""):
    kind, code = SOURCES[src]
    df = load_data(kind, code, years, org)
    if df.empty:
        raise RuntimeError("no data")      # raised (not returned) so that a failure is not cached
    fut, added = pd.DataFrame(), 0
    if kind == "fdorg":
        fut = load_fdorg(code, years, org)[1]
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
            "s_share": s_share, "s_src": s_src, "kind": kind, "elo": fit_elo(df)}


# ---------- automatic feed: injuries, suspensions, absences ----------
FPL_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
ATT_REPLACE = 0.35      # a missing attacker's output is only partly replaced: lose this share of his contribution
CTX_MIN, CTX_MAX = 0.82, 1.10


def _secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def feed_keys():
    prov = st.session_state.get("llm_prov_sel", "Groq (free)")
    name = "GROQ_API_KEY" if prov.startswith("Groq") else "ANTHROPIC_API_KEY"
    return {"tavily": _secret("TAVILY_API_KEY") or st.session_state.get("tav_key", ""),
            "prov": "groq" if prov.startswith("Groq") else "claude",
            "llm_key": _secret(name) or st.session_state.get("llm_key", ""),
            "model": st.session_state.get("llm_model_groq", "openai/gpt-oss-20b") if prov.startswith("Groq")
            else st.session_state.get("llm_model_claude", "claude-sonnet-5-5")}


@st.cache_data(ttl=1800, show_spinner=False)
def load_fpl():
    r = requests.get(FPL_URL, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    js = r.json()
    return js["teams"], js["elements"]


def fpl_context(teams, elements, our_team, now=None):
    """Official Fantasy Premier League data: who is injured/suspended/doubtful and how much the attack depends on him.
    Long absences (news older than 21 days) are ignored: results since then already reflect them."""
    now = now if now is not None else pd.Timestamp.now("UTC").tz_localize(None)
    tinfo = next((t for t in teams if match_team(t["name"], [our_team]) == our_team), None)
    if tinfo is None:
        return None
    pl = [e for e in elements if e.get("team") == tinfo["id"]]
    num = lambda v: float(v) if v not in (None, "") else 0.0
    contrib = {e["id"]: num(e.get("expected_goals")) + num(e.get("expected_assists")) for e in pl}
    tot = sum(contrib.values()) or 1.0
    played = max(int(tinfo.get("played") or 0), 1)
    out = []
    for e in pl:
        stt = e.get("status")
        if stt in ("a", "u", "n", None):
            continue
        chance = e.get("chance_of_playing_next_round")
        avail = chance / 100 if chance is not None else (0.5 if stt == "d" else 0.0)
        try:
            days = (now - pd.Timestamp(e["news_added"]).tz_convert(None)).days
        except Exception:
            days = 0
        long_term = days > 21
        share = contrib[e["id"]] / tot
        pos = POS.get(e.get("element_type"), "")
        regular = num(e.get("starts")) >= 0.5 * played
        att = 0.0 if long_term else share * (1 - avail) * ATT_REPLACE
        dfn = 0.0 if (long_term or not regular) else {"GK": 0.04, "DEF": 0.025}.get(pos, 0.0) * (1 - avail)
        out.append({"name": e.get("web_name", "?"), "pos": pos, "status": {"i": "injured", "s": "suspended", "d": "doubtful"}.get(stt, stt),
                    "chance": chance, "news": e.get("news", ""), "share": round(share, 3), "long_term": long_term,
                    "att": round(att, 4), "dfn": round(dfn, 4)})
    return {"absences": out, "att": max(0.85, 1 - sum(a["att"] for a in out)),
            "dfn": min(1.08, 1 + sum(a["dfn"] for a in out))}


def ctx_multipliers(ctx):
    mh = ctx["home"]["att"] * ctx["away"]["dfn"]
    ma = ctx["away"]["att"] * ctx["home"]["dfn"]
    return float(np.clip(mh, CTX_MIN, CTX_MAX)), float(np.clip(ma, CTX_MIN, CTX_MAX))


def ctx_save(fid, data):
    db_exec("INSERT OR REPLACE INTO context VALUES(?,?,?)", (int(fid), json.dumps(data), dt.datetime.now().isoformat()))


def ctx_load(fid, max_hours=72):
    d = db_read("SELECT data, fetched FROM context WHERE fixture_id=?", (int(fid),))
    if d.empty:
        return None
    if (dt.datetime.now() - dt.datetime.fromisoformat(d.fetched[0])).total_seconds() > max_hours * 3600:
        return None
    return json.loads(d.data[0])


def extract_json(text):
    try:
        a, b = text.index("{"), text.rindex("}")
        return json.loads(text[a:b + 1])
    except Exception:
        return None


def llm_json(keys, prompt):
    """One call to a free (Groq) or paid (Claude) model; expects a JSON answer."""
    if keys["prov"] == "groq":
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=60,
                          headers={"Authorization": f"Bearer {keys['llm_key']}", "Content-Type": "application/json"},
                          json={"model": keys["model"], "temperature": 0,
                                "messages": [{"role": "system", "content": "You extract facts. Reply with JSON only."},
                                             {"role": "user", "content": prompt}]})
        if r.status_code != 200:
            raise RuntimeError(f"AI error {r.status_code}: {r.text[:150]}")
        text = r.json()["choices"][0]["message"]["content"]
    else:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60,
                          headers={"x-api-key": keys["llm_key"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": keys["model"], "max_tokens": 800, "messages": [{"role": "user", "content": prompt}]})
        if r.status_code != 200:
            raise RuntimeError(f"AI error {r.status_code}: {r.text[:150]}")
        text = "".join(b.get("text", "") for b in r.json().get("content", []))
    js = extract_json(text)
    if js is None:
        raise RuntimeError("the AI answer was not readable JSON")
    return js


def player_share(players_df, team, name):
    """Share of the team's attacking output (npxG + xA) from an uploaded Understat players file, if available."""
    if players_df is None or not name:
        return None
    d = players_df
    tm = match_team(team, d.team.unique().tolist())
    if not tm:
        return None
    d = d[d.team == tm]
    val = d.npxg.fillna(d.xg).fillna(0) + d.xa.fillna(0)
    tot = float(val.sum())
    if tot <= 0:
        return None
    last = _key(name).split(" ")[-1] if _key(name) else ""
    hit = d[d.player.astype(str).map(lambda x: bool(last) and last in _key(x))]
    return float(val[hit.index].sum() / tot) if len(hit) else None


def news_context(keys, team, opp, league):
    """Tavily search (free) -> AI extraction (free Groq or paid Claude) -> structured absences with impact."""
    r = requests.post("https://api.tavily.com/search", timeout=40,
                      headers={"Authorization": f"Bearer {keys['tavily']}", "Content-Type": "application/json"},
                      json={"query": f"{team} injury news suspended doubtful ahead of match vs {opp}", "topic": "news",
                            "time_range": "week", "max_results": 5})
    if r.status_code != 200:
        raise RuntimeError(f"Tavily error {r.status_code}")
    snippets = "\n".join(f"- [{x.get('published_date') or ''}] {x.get('title', '')}: {(x.get('content') or '')[:350]}"
                         for x in r.json().get("results", []))
    if not snippets:
        return {"absences": [], "att": 1.0, "dfn": 1.0, "summary": "no recent news found"}
    prompt = (f"Team: {team} ({league}), next opponent {opp}. From the news snippets below, list players of {team} who are "
              "OUT, SUSPENDED or DOUBTFUL for the next match. Ignore injuries older than 3 weeks, players of other teams and "
              'rumours. Reply ONLY with JSON: {"absent":[{"name":str,"role":"GK|DEF|MID|FWD","status":"out|doubtful",'
              '"starter":true|false,"key":true|false}],"summary":str}. If nothing is confirmed reply '
              '{"absent":[],"summary":"nothing confirmed"}.\nSnippets:\n' + snippets)
    js = llm_json(keys, prompt)
    pdf = st.session_state.get("players_df")
    out = []
    for a in (js.get("absent") or [])[:8]:
        if not isinstance(a, dict):
            continue
        avail = 0.0 if str(a.get("status", "out")).lower().startswith("out") else 0.5
        role = str(a.get("role", "")).upper()
        share = player_share(pdf, team, a.get("name", ""))
        att = 0.0
        if role in ("FWD", "MID") and (a.get("key") or share):
            att = (share * ATT_REPLACE if share else 0.04) * (1 - avail)
        dfn = 0.0
        if a.get("starter") and role in ("GK", "DEF"):
            dfn = (0.04 if role == "GK" else 0.025) * (1 - avail)
        out.append({"name": str(a.get("name", "?")), "pos": role, "status": "out" if avail == 0 else "doubtful",
                    "chance": None, "news": "", "share": round(share, 3) if share else None, "long_term": False,
                    "att": round(att, 4), "dfn": round(dfn, 4)})
    return {"absences": out, "att": max(0.85, 1 - sum(a["att"] for a in out)),
            "dfn": min(1.08, 1 + sum(a["dfn"] for a in out)), "summary": str(js.get("summary", ""))[:200]}


def autofeed_one(r, keys):
    if r.league == "Premier League (England)":
        try:
            teams, els = load_fpl()
        except Exception as e:
            return None, f"FPL data unavailable: {e}"
        hc, ac = fpl_context(teams, els, r.home), fpl_context(teams, els, r.away)
        if hc is None or ac is None:
            return None, "team not found in FPL data"
        return {"source": "FPL official data", "home": hc, "away": ac}, "ok"
    if not keys["tavily"] or not keys["llm_key"]:
        return None, "needs free Tavily + AI keys (AI assistant tab)"
    try:
        hc, ac = news_context(keys, r.home, r.away, r.league), news_context(keys, r.away, r.home, r.league)
    except Exception as e:
        return None, str(e)
    return {"source": "news search + AI", "home": hc, "away": ac}, "ok"


def run_feed(fxdf, days, keys, force=False):
    rows, today = [], dt.date.today()
    horizon = (today + dt.timedelta(days=days)).isoformat()
    for r in fxdf.itertuples():
        if r.kickoff and (r.kickoff < today.isoformat() or r.kickoff > horizon):
            continue
        if not force and ctx_load(r.id, 6) is not None:
            rows.append({"Match": f"{r.home} vs {r.away}", "Status": "already fresh"})
            continue
        data, msg = autofeed_one(r, keys)
        row = {"Match": f"{r.home} vs {r.away}", "Status": msg}
        if data:
            ctx_save(r.id, data)
            mh, ma = ctx_multipliers(data)
            row.update({"Source": data["source"], "Home goals x": round(mh, 3), "Away goals x": round(ma, 3),
                        "Absences": len(data["home"]["absences"]) + len(data["away"]["absences"])})
        rows.append(row)
    return rows


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


def analyse_fixture(r, b, w_model=None, w_elo=0.0, ext=None, ctx=None, calib=None):
    """Full output for one stored fixture: ensemble probabilities (Dixon-Coles + Elo sources + market),
    all markets, and the facts needed for the risk check. None if a team is unknown to the model."""
    model = b["model"]
    if r.home not in model["idx"] or r.away not in model["idx"]:
        return None
    neutral = bool(r.neutral)
    lam, mu = lambdas(model, r.home, r.away, neutral)
    lam, mu = lam * float(r.adj_h or 1), mu * float(r.adj_a or 1)
    auto = None
    if ctx:
        mh, ma = ctx_multipliers(ctx)
        lam, mu = lam * mh, mu * ma
        auto = {"mh": mh, "ma": ma, "ctx": ctx}
    m = score_matrix(lam, mu, model["rho"])
    p_dc = np.array(outcome_probs(m))
    sources = {"Dixon-Coles": p_dc}
    elo_list = []
    pe = elo_probs(b.get("elo"), r.home, r.away, neutral)
    if pe is not None:
        sources["Elo (own)"] = np.array(pe)
        elo_list.append(np.array(pe))
    if ext is not None:
        sources["ClubElo"] = np.array(ext)
        elo_list.append(np.array(ext))
    final = p_dc
    w_use = calib["w_elo"] if calib else w_elo
    if elo_list and w_use > 0:
        final = (1 - w_use) * p_dc + w_use * np.mean(elo_list, axis=0)
    if calib and abs(calib["temp"] - 1.0) > 1e-9:
        final = _temp(final, calib["temp"])
    odds3 = [r.o1, r.ox, r.o2]
    if all(o is not None and pd.notna(o) and float(o) > 1 for o in odds3):
        inv = 1 / np.array(odds3, dtype=float)
        sources["Market"] = inv / inv.sum()
        if w_model is not None and w_model < 1:
            final = w_model * final + (1 - w_model) * sources["Market"]
    final = final / final.sum()
    blended = not np.allclose(final, p_dc, atol=1e-9)
    if blended:
        m = tilt_matrix(m, final)
    allp = np.array(list(sources.values()))
    spread = float((allp.max(axis=0) - allp.min(axis=0)).max()) if len(allp) > 1 else 0.0
    mk = masks(m.shape[0])
    market = {"1": r.o1, "X": r.ox, "2": r.o2}
    sels = []
    for k in SEL:
        p = float(m[mk[k]].sum())
        if calib:
            p = _cal_sel(k, p, calib)
        o = market.get(k)
        o = float(o) if o is not None and pd.notna(o) and float(o) > 1 else None
        sels.append({"sel": k, "p": p, "odds": o, "fair": 1 / p if p > 0 else np.inf, "ev": p * o - 1 if o else None})
    flat = np.dstack(np.unravel_index(np.argsort(-m.ravel())[:5], m.shape))[0]
    tops = [(int(i), int(j), float(m[i, j])) for i, j in flat]
    nh, na, grade = data_quality(b, r.home, r.away)
    kd = pd.Timestamp(r.kickoff) if r.kickoff else None
    d = b["df"]

    def rest(t):
        last = d[(d.HomeTeam == t) | (d.AwayTeam == t)].Date.max()
        return int((kd - last).days) if kd is not None and pd.notna(last) else None
    def cong(t):
        if kd is None:
            return None
        return int((((d.HomeTeam == t) | (d.AwayTeam == t)) & (d.Date >= kd - pd.Timedelta(days=14)) & (d.Date < kd)).sum())
    return {"id": int(r.id), "league": r.league, "home": r.home, "away": r.away, "kickoff": r.kickoff,
            "auto": auto, "cong": (cong(r.home), cong(r.away)),
            "lam": lam, "mu": mu, "m": m, "sels": sels, "tops": tops, "nh": nh, "na": na, "grade": grade,
            "adj": (float(r.adj_h or 1), float(r.adj_a or 1)), "note": r.note or "", "b": b, "blended": blended,
            "sources": sources, "spread": spread, "rest": (rest(r.home), rest(r.away)),
            "calibrated": bool(calib) and calib.get("n", 0) >= 120}


def risk_report(f, pick):
    """Risk of a pick, in %. Base = chance the pick fails; extras = reasons to trust the model less."""
    base = (1 - pick["p"]) * 100
    extras, why = 0.0, []
    if f["grade"] == "C":
        extras += 8
        why.append("little recent data for one team (+8)")
    elif f["grade"] == "B":
        extras += 3
        why.append("moderate recent data (+3)")
    if f["spread"] > 0.10:
        x = min(15.0, (f["spread"] - 0.08) * 100 * 0.8)
        extras += x
        why.append(f"sources disagree by {f['spread'] * 100:.0f} points (+{x:.0f})")
    if any(r is not None and r <= 3 for r in f["rest"]):
        extras += 4
        why.append("a team has 3 days or less of rest (+4)")
    if f["adj"] != (1.0, 1.0):
        extras += 3
        why.append("manual adjustment applied, news-driven (+3)")
    au = f.get("auto")
    if au and (abs(au["mh"] - 1) > 0.005 or abs(au["ma"] - 1) > 0.005):
        extras += 3
        why.append("injury/absence news applied (+3)")
    if any(c is not None and c >= 4 for c in f.get("cong", (None, None))):
        extras += 3
        why.append("4+ matches in the last 14 days for a team (+3)")
    total = min(95.0, base + extras)
    return {"risk": total, "base": base, "extra": extras, "why": why,
            "label": "Low" if total <= 25 else "Medium" if total <= 40 else "High",
            "safe_p": max(0.0, pick["p"] - extras / 100)}


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
    if f.get("blended"):
        s += f" The 1X2 probabilities combine {len(f['sources'])} sources ({', '.join(f['sources'])})."
    if f.get("spread", 0) > 0.08:
        s += f" The sources disagree by up to {f['spread'] * 100:.0f} points, so the call is less reliable."
    if f["adj"] != (1.0, 1.0):
        s += f" Manual adjustment applied (x{f['adj'][0]:.2f} / x{f['adj'][1]:.2f})."
    if f.get("auto"):
        s += (f" Injury/absence feed ({f['auto']['ctx']['source']}) changed expected goals by "
              f"x{f['auto']['mh']:.2f} / x{f['auto']['ma']:.2f}.")
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


# ---------- free news headlines (no key needed) ----------
@st.cache_data(ttl=1800, show_spinner=False)
def team_news(team, days=10, n=6):
    q = requests.utils.quote(f'"{team}" football (injury OR injured OR suspended OR lineup OR "ruled out")')
    r = requests.get(f"https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en", timeout=15,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return parse_news(r.content, days, n)


def parse_news(xml_bytes, days=10, n=6):
    items, cut = [], pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=days)
    for it in ET.fromstring(xml_bytes).iter("item"):
        try:
            d = pd.Timestamp(parsedate_to_datetime(it.findtext("pubDate"))).tz_convert(None)
        except Exception:
            continue
        if d >= cut:
            items.append({"date": d, "title": it.findtext("title") or "", "link": it.findtext("link") or "",
                          "source": it.findtext("source") or ""})
    return sorted(items, key=lambda x: x["date"], reverse=True)[:n]


# ---------- Tavily: free AI search engine (1,000 credits a month, no card) ----------
def tavily_research(key, f):
    """One short search per topic (1 credit each). Returns a list of {topic, answer, results}."""
    topics = [(f"{f['home']} injuries and suspensions", f"{f['home']} football team news injuries suspended doubtful"),
              (f"{f['away']} injuries and suspensions", f"{f['away']} football team news injuries suspended doubtful"),
              ("Expected line-ups and context", f"{f['home']} vs {f['away']} predicted lineup preview")]
    out, err = [], None
    for name, q in topics:
        try:
            r = requests.post("https://api.tavily.com/search", timeout=40,
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"query": q, "topic": "news", "time_range": "week", "max_results": 5,
                                    "include_answer": "basic"})
        except Exception as e:
            err = f"Network error: {e}"
            break
        if r.status_code != 200:
            err = f"Tavily error {r.status_code}: {r.text[:200]}"
            break
        js = r.json()
        out.append({"topic": name, "answer": js.get("answer") or "", "results": js.get("results", [])})
    return out, err


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
    if SOURCES[league][0] == "fdorg" and not org_key():
        st.info("This competition needs a free football-data.org key: register at football-data.org/client/register, "
                "copy the key from the email and paste it below (or save it as FOOTBALL_DATA_ORG_KEY in Secrets).")
        st.text_input("football-data.org key", type="password", key="fdorg_key")
    b = league_bundle(league, years, half_life, wx_user, fuse_on)
    if b is None:
        st.error("Could not load data for this league (check the internet connection of the host and any key it needs).")
        _kd, _cd = SOURCES[league]
        _why = net_failures(f"{_cd}-") if _kd == "fd" else net_failures(f"new-{_cd}") if _kd == "new" else []
        if _why:
            st.caption("Reason: " + " | ".join(_why[:3]))
        if st.button("Try again now", key=f"retry_{league}"):
            HOST_DEAD.clear()
            st.cache_data.clear()
            st.rerun()
        st.caption("Open the 'Data & backup' tab and press 'Test all sources now' to see exactly what is blocked.")
    else:
        _kd, _cd = SOURCES[league]
        if any(v[0] == "saved copy" for k, v in NET_LOG.items() if k.startswith(f"{_cd}-") or k == f"new-{_cd}"):
            st.warning("The internet source did not answer: using the last saved copy of the data (may be a bit old).")
        if _cd in BACKUP_USED:
            st.warning("Main source unreachable: this league uses the backup source (openfootball), "
                       "so shots/xG and bookmaker odds are not available.")
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
        fd_leagues = [k for k in BOARD_LEAGUES if SOURCES[k][0] in ("fd", "fdorg")]
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
    with st.expander("Settings", expanded=True):
        c = st.columns(3)
        allowed = c[0].multiselect("Markets to consider", SEL, default=DEFAULT_MARKETS, key="an_mk")
        min_p = c[1].slider("Minimum probability %", 40, 90, 60, 1, key="an_p") / 100
        min_fair = c[2].slider("Minimum fair odds (skips trivial picks)", 1.05, 2.5, 1.25, 0.05, key="an_o")
        c = st.columns(3)
        w_elo = c[0].slider("Weight of the Elo sources %", 0, 60, 20, 5, key="an_elo") / 100
        use_ext = c[1].checkbox("Include ClubElo (free external ratings)", value=True, key="an_ext")
        w_model = c[2].slider("Weight of our model vs market odds % (only if odds exist)", 0, 100, 50, 5, key="an_w") / 100
        st.caption("Sources combined: Dixon-Coles model + own Elo + ClubElo + market odds (when given). Odds are "
                   "optional; without them the first three are used. Check the Backtest page (Elo and Blend rows) "
                   "to see which weights work best for a league.")
        use_cal = st.checkbox("Use each league's calibrated settings (best Elo weight + honest confidence)", value=True,
                              key="an_cal")
        need_agree = st.checkbox("Safe slip: only picks where all sources agree", value=True, key="an_agree")
        if use_cal:
            _missing = [lg for lg in sorted(fxdf.league.unique()) if calib_load(lg) is None]
            if _missing:
                st.info("Not calibrated yet: " + ", ".join(_missing) + ". Calibration tests the model on each league's "
                        "latest matches and keeps the settings that predicted best (about a minute per league, done once).")
                if st.button("Calibrate these leagues now", key="cal_now"):
                    for lg in _missing:
                        b0 = league_bundle(lg, years, half_life, wx_user, fuse_on)
                        if b0 is None:
                            st.warning(f"{lg}: no data")
                            continue
                        with st.spinner(f"Calibrating {lg}..."):
                            run_calibration(lg, b0, *eff_params(lg, half_life, wx_user))
                    st.rerun()
        use_auto = st.checkbox("Apply the automatic injury/absence feed (Premier League: official data; other leagues: "
                               "needs the free keys in the AI assistant tab)", value=True, key="an_auto")
        c = st.columns(2)
        max_risk = c[0].slider("Maximum risk per match %", 10, 60, 30, 1, key="an_risk")
        target = c[1].slider("Safe slip: minimum combined chance % (stress-tested)", 30, 90, 55, 1, key="an_target") / 100
    if use_auto and not st.session_state.get("feed_ran"):
        st.session_state["feed_ran"] = True
        with st.spinner("Updating injury and absence data for the next 3 days..."):
            try:
                run_feed(fxdf, 3, feed_keys())
            except Exception:
                pass
    fxs, skipped, bundles, calibs = [], [], {}, {}
    for r in fxdf.itertuples():
        if r.league not in bundles:
            bundles[r.league] = league_bundle(r.league, years, half_life, wx_user, fuse_on)
        b = bundles[r.league]
        f = None
        if b:
            ext = clubelo_probs(r.league, r.home, r.away, bool(r.neutral), b["elo"]["d0"]) if use_ext else None
            if r.league not in calibs:
                calibs[r.league] = calib_load(r.league) if use_cal else None
            f = analyse_fixture(r, b, w_model, w_elo, ext, ctx_load(r.id) if use_auto else None, calibs[r.league])
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
        rk = risk_report(f, bp) if bp else None
        ag = agreement(f, bp) if bp else "-"
        rows.append({"League": f["league"], "Date": f["kickoff"], "Match": f"{f['home']} vs {f['away']}",
                     "Goals exp.": f"{f['lam']:.2f}-{f['mu']:.2f}", "Likely score": f"{f['tops'][0][0]}-{f['tops'][0][1]}",
                     "1 %": round(P["1"] * 100), "X %": round(P["X"] * 100), "2 %": round(P["2"] * 100),
                     "Over 2.5 %": round(P["Over 2.5"] * 100), "BTTS %": round(P["BTTS Yes"] * 100),
                     "Best pick": bp["sel"] if bp else "-", "Pick %": round(bp["p"] * 100) if bp else None,
                     "Risk %": round(rk["risk"]) if rk else None, "Risk": rk["label"] if rk else "-",
                     "Agreement": ag, "Verdict": verdict(rk, ag, f) if rk else "-",
                     "Calibrated": "yes" if f.get("calibrated") else "no",
                     "Sources": len(f["sources"]), "Data": f["grade"]})
    st.markdown("#### Predictions for all matches")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Risk % = chance the pick fails plus penalties for weak data, disagreeing sources, short rest or "
               "news-driven adjustments. Low <= 25, Medium <= 40, High above. Data grade: A = 30+ recent matches "
               "for both teams, B = 18+, C = fewer. Verdict: Strong = low risk, sources agree, enough data; "
               "OK = medium risk and no disagreement; Skip = anything else. 'Calibrated' = the league's settings were "
               "tested on its recent matches (Analysis settings or Data tab).")

    legs = top_legs(fxs, allowed, min_p, min_fair)
    for l in legs:
        l["risk"] = risk_report(fxs[l["fi"]], l)
        l["agree"] = agreement(fxs[l["fi"]], l)
        l["verdict"] = verdict(l["risk"], l["agree"], fxs[l["fi"]])
    st.markdown("#### Pick quality and risk")
    if not legs:
        st.warning("No selection passes the filters. Lower the minimum probability or fair odds.")
    else:
        st.dataframe(pd.DataFrame([{"League": l["league"], "Match": l["match"], "Pick": l["sel"],
                                    "Model %": round(l["p"] * 100, 1), "Risk %": round(l["risk"]["risk"]),
                                    "Risk": l["risk"]["label"], "Stress-tested %": round(l["risk"]["safe_p"] * 100, 1),
                                    "Agreement": l["agree"], "Verdict": l["verdict"],
                                    "Why": "; ".join(l["risk"]["why"]) or "-"} for l in legs]),
                     hide_index=True, use_container_width=True)
        ok = sorted([l for l in legs if l["risk"]["risk"] <= max_risk and not (need_agree and l["agree"] == "split")],
                    key=lambda l: (l["risk"]["risk"], -l["p"]))
        out = [l for l in legs if l["risk"]["risk"] > max_risk or (need_agree and l["agree"] == "split")]
        safe, comb = [], 1.0
        for l in ok:
            if safe and comb * l["risk"]["safe_p"] < target:
                break
            safe.append(l)
            comb *= l["risk"]["safe_p"]
        st.markdown("#### Recommended safe slip")
        if not safe:
            st.warning(f"No match is below {max_risk}% risk. Raise the limit or lower the probability filters.")
        else:
            raw = float(np.prod([l["p"] for l in safe]))
            st.dataframe(pd.DataFrame([{"League": l["league"], "Match": l["match"], "Pick": l["sel"],
                                        "Model %": round(l["p"] * 100, 1), "Risk %": round(l["risk"]["risk"])}
                                       for l in safe]), hide_index=True, use_container_width=True)
            dist = simulate(fxs, safe)
            c = st.columns(3)
            c[0].metric("Chance every pick wins", f"{raw * 100:.1f}%")
            c[1].metric("Stress-tested chance", f"{comb * 100:.1f}%")
            c[2].metric("Chance at most one fails", f"{(dist[:2].sum()) * 100:.1f}%")
            if comb < target:
                st.warning("Even the safest single pick is below your target. No slip meets it.")
            st.caption("Built greedily from the lowest-risk picks while the stress-tested combined chance stays above "
                       "your target. Stress-tested = each pick's probability minus its model-risk penalties. "
                       "No outcome is guaranteed: even a 70% slip fails 3 times in 10.")
        if out:
            with st.expander(f"Excluded as too risky ({len(out)})"):
                for l in out:
                    st.write(f"- **{l['match']}** ({l['sel']}, risk {l['risk']['risk']:.0f}%): "
                             + ("; ".join(l["risk"]["why"] + (["sources disagree on the result"] if l["agree"] == "split" else []))
                                or f"only {l['p'] * 100:.0f}% likely"))
        with st.expander("Build your own slip"):
            names = [l["match"] for l in legs]
            chosen_names = st.multiselect("Matches in the slip", names, default=[l["match"] for l in safe], key="an_slip")
            chosen = [l for l in legs if l["match"] in chosen_names]
            if chosen:
                sizes = []
                for k in range(1, len(chosen) + 1):
                    s2 = slip_stats(chosen[:k])
                    sizes.append({"Legs": k, "Chance all win %": round(s2["P"] * 100, 1),
                                  "Chance slip fails %": round((1 - s2["P"]) * 100, 1),
                                  "Combined fair odds": round(s2["fair"], 2)})
                st.dataframe(pd.DataFrame(sizes), hide_index=True, use_container_width=True)
                s2 = slip_stats(chosen)
                d2 = simulate(fxs, chosen)
                c = st.columns(3)
                c[0].metric("Chance every pick wins", f"{s2['P'] * 100:.1f}%")
                c[1].metric("Exactly one fails", f"{(d2[1] * 100 if len(d2) > 1 else 0):.1f}%")
                c[2].metric("Two or more fail", f"{d2[2:].sum() * 100:.1f}%")
                st.caption(f"Monte Carlo with {N_SIMS:,} rounds; matches treated as independent.")

    tbl = pd.DataFrame(rows)
    st.download_button("⬇️ Download predictions (CSV)", tbl.to_csv(index=False), "predictions.csv", "text/csv", key="an_dl")
    if legs:
        st.text_area("Copy-ready summary", "\n".join(
            f"{i + 1}. {l['match']} ({l['league']}): {l['sel']} - {l['p'] * 100:.0f}% | risk {l['risk']['risk']:.0f}% ({l['risk']['label']})"
            for i, l in enumerate(legs)), height=120, key="an_copy")

    st.markdown("#### Match detail")
    labels = [f"{f['league']} | {f['home']} vs {f['away']} | {f['kickoff']}" for f in fxs]
    sel = st.selectbox("Choose a match", labels, key="an_detail")
    f = fxs[labels.index(sel)]
    st.info(explain(f))
    with st.expander("How reliable has this league's model been? (measured on its recent matches)"):
        render_reliability(f["league"], calibs.get(f["league"]) or calib_load(f["league"]))
    c = st.columns(3)
    c[0].metric(f"{f['home']} expected goals", f"{f['lam']:.2f}")
    c[1].metric(f"{f['away']} expected goals", f"{f['mu']:.2f}")
    c[2].metric("Data grade", f["grade"], f"{f['nh']} / {f['na']} recent matches", delta_color="off")
    st.markdown("**What each source says (home / draw / away %)**")
    st.dataframe(pd.DataFrame([{"Source": k, "Home %": round(v[0] * 100, 1), "Draw %": round(v[1] * 100, 1),
                                "Away %": round(v[2] * 100, 1)} for k, v in f["sources"].items()]),
                 hide_index=True, use_container_width=True)
    if f["auto"]:
        st.markdown(f"**Absences from the automatic feed** ({f['auto']['ctx']['source']}): goals x{f['auto']['mh']:.2f} "
                    f"for {f['home']}, x{f['auto']['ma']:.2f} for {f['away']}")
        rows_a = [{"Team": f[side], **{k: a[k] for k in ("name", "pos", "status", "chance", "share", "long_term", "news")}}
                  for side, key in (("home", "home"), ("away", "away")) for a in f["auto"]["ctx"][key]["absences"]]
        if rows_a:
            st.dataframe(pd.DataFrame(rows_a).rename(columns={"name": "Player", "pos": "Pos", "status": "Status",
                                                              "chance": "Chance of playing %", "share": "Share of attack",
                                                              "long_term": "Long-term (ignored)", "news": "News"}),
                         hide_index=True, use_container_width=True)
        else:
            st.write("No relevant absences found.")
        st.caption("Impact is a cautious estimate: a missing attacker removes about a third of his share of the "
                   "team's attack; missing regular defenders/keeper raise the opponent's goals slightly. Capped at "
                   "about -18% / +10%.")
    cc = st.columns(2)
    for col, tm, rs in zip(cc, (f["home"], f["away"]), f["rest"]):
        d5 = last_matches(f["b"]["df"], tm, 5)
        if d5.empty:
            continue
        col.markdown(f"**{tm}**: form {' '.join(d5.Res)} | {d5.GF.sum()} scored, {d5.GA.sum()} conceded | "
                     + (f"{rs} days since last match" + (" ⚠️ short rest" if rs <= 3 else "") if rs is not None else ""))
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
    st.markdown("#### 🤖 Smart research assistant")
    st.caption("The model knows results, not news. Pick a free option below to look for injuries, suspensions and "
               "line-up news, then adjust the match yourself.")
    fxdf = db_read("SELECT * FROM fixtures ORDER BY kickoff, league, id")
    if fxdf.empty:
        st.info("Add matches first.")
        return
    st.markdown("##### ⚡ Automatic injury and absence feed")
    st.caption("Premier League matches need nothing: the official Fantasy Premier League data lists injured, suspended "
               "and doubtful players. For other leagues the app searches the news (Tavily, free) and a free AI (Groq) "
               "reads the headlines and lists the absences. Everything is stored and used automatically in the Analysis tab.")
    _fk = feed_keys()
    st.caption(f"Keys found: Tavily {'✅' if _fk['tavily'] else '❌ not found'} | "
               f"AI ({_fk['prov']}) {'✅' if _fk['llm_key'] else '❌ not found'}. "
               "Keys saved in the secrets.toml file are loaded when the app starts.")
    with st.expander("Free keys for the other leagues (optional)"):
        st.markdown("- **Tavily** (search): sign up at tavily.com, copy the key starting with `tvly-` (1,000 free searches a month)\n"
                    "- **Groq** (free AI): sign up at console.groq.com, API Keys, Create key (no card)")
        st.text_input("Tavily key (kept only in this session)", type="password", key="tav_key")
        st.radio("AI that reads the news", ["Groq (free)", "Claude (paid)"], horizontal=True, key="llm_prov_sel")
        st.text_input("AI key (kept only in this session)", type="password", key="llm_key")
        _g = st.session_state.get("llm_prov_sel", "Groq (free)").startswith("Groq")
        st.text_input("AI model (change if the provider renames it)", "openai/gpt-oss-20b" if _g else "claude-sonnet-5-5",
                      key="llm_model_groq" if _g else "llm_model_claude")
    c = st.columns(2)
    days_ahead = c[0].slider("Matches in the next N days", 1, 10, 3, key="feed_days")
    force = c[1].checkbox("Refresh even if recent", value=False, key="feed_force")
    if st.button("Run the automatic feed now", key="feed_go"):
        with st.spinner("Collecting injury and absence data..."):
            rows_f = run_feed(fxdf, days_ahead, feed_keys(), force)
        if rows_f:
            st.dataframe(pd.DataFrame(rows_f), hide_index=True, use_container_width=True)
        else:
            st.info("No matches in that window.")
    st.markdown("##### Match research and manual adjustment")
    labels = [f"#{r.id} {r.league} | {r.home} vs {r.away} | {r.kickoff}" for r in fxdf.itertuples()]
    sel = st.selectbox("Match", labels, key="ai_match")
    r = list(fxdf.itertuples())[labels.index(sel)]
    f = {"home": r.home, "away": r.away, "league": r.league, "kickoff": r.kickoff}
    store = st.session_state.setdefault("ai_reports", {})
    prov = st.radio("Source", ["📰 Free headlines (no key)", "🔎 Tavily AI search (free key, 1,000 searches a month)",
                               "🧠 Claude with web search (paid key)"], key="ai_prov")

    if prov.startswith("📰"):
        if st.button("Fetch latest headlines for both teams", key=f"news_{r.id}"):
            for tm in (r.home, r.away):
                st.markdown(f"**{tm}**")
                try:
                    items = team_news(tm)
                except Exception as e:
                    st.warning(f"Could not fetch news: {e}")
                    continue
                if not items:
                    st.write("No recent headlines about injuries or line-ups.")
                for x in items:
                    st.markdown(f"- {x['date']:%d %b} [{x['title']}]({x['link']})")
            st.caption("Headlines only, from Google News. Open the links to verify before adjusting anything.")
    elif prov.startswith("🔎"):
        st.caption("Free key without a card: sign up at tavily.com, copy the key that starts with `tvly-`. "
                   "Each match uses 3 of your 1,000 monthly credits.")
        tkey = feed_keys()["tavily"]
        if not tkey:
            st.info("No Tavily key found. Save it in secrets.toml (then restart the app) or paste it in the "
                    "'Free keys' box in the AI assistant tab.")
        if st.button("Search the web for this match", key="tav_go"):
            if not tkey:
                st.error("Enter your Tavily key first.")
            else:
                with st.spinner("Searching..."):
                    res, err = tavily_research(tkey, f)
                if err:
                    st.error(err)
                if res:
                    store[("tav", int(r.id))] = res
        for blk in store.get(("tav", int(r.id)), []):
            st.markdown(f"**{blk['topic']}**")
            if blk["answer"]:
                st.write(blk["answer"])
            for x in blk["results"][:4]:
                st.markdown(f"- [{x.get('title', 'source')}]({x.get('url', '')}) {x.get('published_date') or ''}")
        if store.get(("tav", int(r.id))):
            st.caption("Summaries are machine-generated from web pages. Open the sources before relying on them.")
    else:
        key = _secret("ANTHROPIC_API_KEY") or st.text_input("Anthropic API key (kept only in this session)", type="password", key="ai_key")
        c = st.columns(2)
        model_id = c[0].text_input("Model", "claude-sonnet-5-5", key="ai_model")
        max_uses = c[1].slider("Max web searches per match", 1, 8, 4, key="ai_uses")
        if st.button("Search the web for this match", key="ai_go"):
            if not key:
                st.error("Enter an API key first.")
            else:
                with st.spinner("Searching..."):
                    txt, cites, err = ai_research(key, model_id, f, max_uses)
                if err:
                    st.error(err)
                else:
                    store[("cl", int(r.id))] = (txt, cites)
        if ("cl", int(r.id)) in store:
            txt, cites = store[("cl", int(r.id))]
            st.markdown(txt)
            if cites:
                st.markdown("**Sources:** " + " | ".join(f"[{t}]({u})" for u, t in cites.items()))
            st.caption("This text comes from web pages and a language model. Check it before relying on it.")

    st.markdown("**Adjust the model for this match** (your judgement after reading the news)")
    st.caption("Multiplies a team's expected goals. Example: main striker out about 0.92; weakened rotation 0.9. "
               "An adjusted match gets a small risk penalty because it is news-driven.")
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
    with st.expander("Connection check (press if a league fails to load)"):
        if st.button("Test all sources now", key="conn_btn"):
            HOST_DEAD.clear()
            tests = [("football-data.co.uk (Serie A, this season)", f"{FD}/mmz4281/{_season_tag()}/I1.csv"),
                     ("football-data.co.uk (Serie A, last season)", f"{FD}/mmz4281/{_season_tag(-1)}/I1.csv"),
                     ("football-data.co.uk (fixtures)", f"{FD}/fixtures.csv"),
                     ("openfootball (GitHub)", f"{OFB}/{_season_dash()}/it.1.json"),
                     ("International results (GitHub)", INTL_URL),
                     ("ClubElo", "http://api.clubelo.com/" + str(dt.date.today())),
                     ("FPL official API", FPL_URL),
                     ("Google News RSS", "https://news.google.com/rss/search?q=football&hl=en")]
            rows = []
            for name, url in tests:
                t0 = time.time()
                raw, err = fetch_bytes(url, tries=1, timeout=15)
                rows.append({"Source": name, "Result": "OK" if raw else "FAILED", "Detail": err or f"{len(raw)//1024} KB",
                             "Seconds": round(time.time() - t0, 1)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("A 404 for 'this season' is normal in the first weeks of a new season; the app then uses "
                       "the previous seasons. If everything FAILED, the problem is your internet / firewall / antivirus.")
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
               "fused by team-name matching), martj42 international results (national teams), ClubElo (external ratings), "
               "FPL (Premier League injuries).")
    st.markdown("**Auto-tune a league**")
    st.caption("Tries a few settings (recency half-life and xG weight) and keeps the one that predicted the most "
               "recent matches best. It only switches away from the default when the gain is clear, so it does not "
               "chase noise. Takes about 30-60 seconds per league.")
    tl = st.multiselect("Leagues to tune", BOARD_LEAGUES, default=[], key="tune_leagues")
    if st.button("Tune selected leagues", key="tune_btn") and tl:
        for lg in tl:
            try:
                b = _league_bundle(lg, years, 300, 0.5, fuse_on)
            except RuntimeError:
                b = None
            if b is None:
                st.warning(f"{lg}: no data")
                continue
            d = b["df"]
            has_xg = not b["xg"].startswith("not available")
            grid = [(h, w) for h in (150, 300, 600) for w in ((0.0, 0.5) if has_xg else (0.0,))]
            res = []
            with st.spinner(f"Tuning {lg}..."):
                for h, w in grid:
                    ll, n = wf_logloss(d, float(np.log(2) / h), w)
                    res.append({"Half-life": h, "xG weight": w, "Log loss": round(ll, 4), "Matches": n})
            t = pd.DataFrame(res)
            dflt = t[(t["Half-life"] == 300) & (t["xG weight"] == (0.5 if has_xg else 0.0))]
            base = float(dflt["Log loss"].iloc[0]) if len(dflt) else float(t["Log loss"].max())
            best = t.loc[t["Log loss"].idxmin()]
            switched = base - float(best["Log loss"]) >= 0.002
            use = best if switched or not len(dflt) else dflt.iloc[0]
            db_exec("INSERT OR REPLACE INTO league_params VALUES(?,?,?,?,?,?)",
                    (lg, int(use["Half-life"]), float(use["xG weight"]), float(use["Log loss"]), base,
                     dt.datetime.now().isoformat()))
            st.markdown(f"**{lg}** -> half-life {int(use['Half-life'])} days, xG weight {float(use['xG weight'])}"
                        + (" (changed)" if switched else " (default kept)"))
            st.dataframe(t, hide_index=True, use_container_width=True)
    st.markdown("**Calibrate confidence (recommended)**")
    st.caption("Tests the model on each league's latest ~300 matches, one forecast at a time without peeking at the "
               "result, then keeps the Elo weight and confidence correction that predicted best. It also measures how "
               "often the model's 40%, 50%, 60%... forecasts really happen. Do this after 'Tune' (if you use it).")
    cl = st.multiselect("Leagues to calibrate", BOARD_LEAGUES, default=[], key="cal_leagues")
    if st.button("Calibrate selected leagues", key="cal_btn") and cl:
        for lg in cl:
            b0 = league_bundle(lg, years, half_life, wx_user, fuse_on)
            if b0 is None:
                st.warning(f"{lg}: no data")
                continue
            with st.spinner(f"Calibrating {lg}..."):
                res, info = run_calibration(lg, b0, *eff_params(lg, half_life, wx_user))
            st.markdown(f"**{lg}**: Elo weight {res['w_elo'] * 100:.0f}%, temperature {res['temp']:.2f}, "
                        f"{res['n']} matches tested")
            render_reliability(lg, calib_load(lg))
    cal_all = db_read("SELECT league, w_elo, temp, pa, pb, ba, bb, loss, base_loss, n, tuned_at FROM league_calib")
    if len(cal_all):
        st.dataframe(cal_all, hide_index=True, use_container_width=True)
        if st.button("Reset all calibration", key="cal_reset"):
            db_exec("DELETE FROM league_calib")
            st.success("Calibration cleared.")
    tuned = db_read("SELECT league, half_life, wx, loss, base_loss, tuned_at FROM league_params")
    if len(tuned):
        st.dataframe(tuned, hide_index=True, use_container_width=True)
        if st.button("Reset all tuning", key="tune_reset"):
            db_exec("DELETE FROM league_params")
            st.success("Tuning cleared.")
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
st.sidebar.caption(f"Version {APP_VERSION}")
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
    if kind == "fdorg" and not org_key():
        sb.text_input("football-data.org free key", type="password", key="fdorg_key")
        st.info("Paste your free football-data.org key in the sidebar (register at football-data.org/client/register).")
        st.stop()
    df = load_data(kind, code, years, org_key() if kind == "fdorg" else "")
    if fuse_on and kind == "fd" and not df.empty:
        _extra, _ = load_openfootball(code, years)
        if not _extra.empty:
            df, _added = fuse(df, [_extra])
            sb.caption(f"Fusion: +{_added} matches from the second source")
    if code in BACKUP_USED:
        sb.warning("Main source was unreachable: using the backup source (openfootball).")

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
