import numpy as np
import pandas as pd
import streamlit as st

SEL = ["1", "X", "2", "1X", "X2", "12", "Over 1.5", "Over 2.5", "Under 2.5", "Under 3.5", "BTTS Yes", "BTTS No"]
N_SIMS = 20000


# ======================= SLIP ENGINE =======================
def masks(n):
    i, j = np.indices((n, n))
    t = i + j
    return {"1": i > j, "X": i == j, "2": i < j, "1X": i >= j, "X2": i <= j, "12": i != j,
            "Over 1.5": t > 1, "Over 2.5": t > 2, "Under 2.5": t < 3, "Under 3.5": t < 4,
            "BTTS Yes": (i > 0) & (j > 0), "BTTS No": (i == 0) | (j == 0)}


def evaluate(rows, mat_fn):
    """Turn the fixtures table into per-match selections with model probability, fair odds and edge."""
    fx = []
    for _, r in rows.iterrows():
        h, a = r.get("Home"), r.get("Away")
        if not isinstance(h, str) or not isinstance(a, str) or h == a:
            continue
        m = mat_fn(h, a)
        mk = masks(m.shape[0])
        sels = []
        for k in SEL:
            p = float(m[mk[k]].sum())
            o = r.get(k)
            o = float(o) if pd.notna(o) and float(o) > 1 else None
            sels.append({"sel": k, "p": p, "odds": o, "fair": 1 / p if p > 0 else np.inf,
                         "ev": p * o - 1 if o else None})
        fx.append({"home": h, "away": a, "m": m, "sels": sels})
    return fx


def pick_legs(fx, mode, min_prob, min_odds, min_edge):
    legs = []
    for fi, f in enumerate(fx):
        best = None
        for s in f["sels"]:
            eff = s["odds"] or s["fair"]
            if s["p"] < min_prob or eff < min_odds:
                continue
            if mode == "value":
                if s["ev"] is None or s["ev"] < min_edge:
                    continue
                score = s["ev"]
            else:
                score = s["p"]
            if best is None or score > best[0]:
                best = (score, s)
        if best:
            legs.append({**best[1], "match": f"{f['home']} vs {f['away']}", "fi": fi, "score": best[0]})
    return sorted(legs, key=lambda l: -l["score"])


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


def arbitrage(odds):
    o = odds.where(odds > 1)
    best = o.max()
    if best.isna().any():
        return None
    s = float((1 / best).sum())
    return {"best": best, "book": o.idxmax(), "margin": s, "profit_pct": (1 / s - 1) * 100, "w": (1 / best) / s}


# ======================= PLAYERS =======================
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


# ======================= UI =======================
def render_slip_tab(model, teams, neutral, lambdas, score_matrix):
    st.info("No bet is guaranteed. This tool ranks selections by the model's probability and edge, and shows how "
            "fast a slip's risk grows with every extra leg. Only stake money you can afford to lose.")
    c = st.columns(2)
    neu = c[0].checkbox("Neutral venues", value=neutral, key="slip_neu")
    label = c[1].radio("Strategy", ["Safest (highest probability)", "Value (best edge vs odds)"], key="slip_mode")
    mode = "value" if label.startswith("Value") else "safe"
    c = st.columns(3)
    min_prob = c[0].slider("Min probability per leg %", 40, 95, 60, 1, key="slip_p") / 100
    min_odds = c[1].slider("Min odds per leg", 1.05, 3.0, 1.20, 0.05, key="slip_o")
    min_edge = c[2].slider("Min edge % (value mode)", 0, 20, 3, 1, key="slip_e") / 100

    st.markdown("**Fixtures** (add rows; odds are optional but needed for value mode)")
    base = pd.DataFrame({"Home": [None] * 3, "Away": [None] * 3, **{k: [np.nan] * 3 for k in SEL}})
    cfg = {"Home": st.column_config.SelectboxColumn("Home", options=teams),
           "Away": st.column_config.SelectboxColumn("Away", options=teams),
           **{k: st.column_config.NumberColumn(k, min_value=1.01, max_value=200.0, format="%.2f") for k in SEL}}
    rows = st.data_editor(base, column_config=cfg, num_rows="dynamic", hide_index=True,
                          use_container_width=True, key="slip_rows")
    fx = evaluate(rows, lambda h, a: score_matrix(*lambdas(model, h, a, neu), model["rho"]))
    if not fx:
        st.write("Pick at least one fixture above.")
        return
    for f in fx:
        with st.expander(f"All selections: {f['home']} vs {f['away']}"):
            st.dataframe(pd.DataFrame([{"Selection": s["sel"], "Model %": round(s["p"] * 100, 1),
                                        "Fair odds": round(s["fair"], 2), "Your odds": s["odds"],
                                        "Edge %": None if s["ev"] is None else round(s["ev"] * 100, 1)}
                                       for s in f["sels"]]), hide_index=True, use_container_width=True)
    legs = pick_legs(fx, mode, min_prob, min_odds, min_edge)
    if not legs:
        st.warning("No selection passes your filters. Lower the minimum probability/odds, or enter odds.")
        return
    st.markdown("**Suggested legs** (best first)")
    st.dataframe(pd.DataFrame([{"Match": l["match"], "Pick": l["sel"], "Model %": round(l["p"] * 100, 1),
                                "Odds": l["odds"] or f"{l['fair']:.2f}*",
                                "Edge %": None if l["ev"] is None else round(l["ev"] * 100, 1)} for l in legs]),
                 hide_index=True, use_container_width=True)
    st.caption("* no odds entered: fair odds shown instead.")
    sizes = []
    for k in range(1, len(legs) + 1):
        s = slip_stats(legs[:k])
        sizes.append({"Legs": k, "Chance all win %": round(s["P"] * 100, 1), "Chance slip fails %": round((1 - s["P"]) * 100, 1),
                      "Combined odds": round(s["O"], 2), "EV %": None if s["ev"] is None else round(s["ev"] * 100, 1)})
    st.markdown("**How risk grows with slip size**")
    st.dataframe(pd.DataFrame(sizes), hide_index=True, use_container_width=True)
    k = st.slider("Legs in your slip", 1, len(legs), min(len(legs), 3), key="slip_k") if len(legs) > 1 else 1
    chosen = legs[:k]
    s = slip_stats(chosen)
    dist = simulate(fx, chosen)
    c = st.columns(3)
    c[0].metric("Chance all legs win", f"{s['P'] * 100:.1f}%")
    c[1].metric("Combined odds" + ("" if s["real_odds"] else " (fair)"), f"{s['O']:.2f}")
    c[2].metric("Expected value", "n/a" if s["ev"] is None else f"{s['ev'] * 100:+.1f}%")
    st.write(f"Simulation ({N_SIMS:,} runs): all win **{dist[0] * 100:.1f}%** | exactly one leg loses "
             f"**{(dist[1] * 100 if len(dist) > 1 else 0):.1f}%** | two or more lose **{dist[2:].sum() * 100:.1f}%**")
    bank = st.number_input("Bankroll", 1.0, 1e9, 1000.0, key="slip_bank")
    if s["real_odds"] and s["ev"] is not None and s["ev"] > 0:
        stake = bank * min(0.05, 0.25 * s["kelly"])
        st.success(f"Positive edge on your odds. Quarter-Kelly stake (capped at 5%): **{stake:,.0f}**")
    elif s["real_odds"]:
        st.error("No edge on these odds: the model expects to lose money on this slip. Do not stake.")
    else:
        st.info("Enter bookmaker odds for every leg to see expected value and a stake suggestion.")
    st.caption("Legs are treated as independent. The model ignores injuries and line-ups, so a large edge "
               "usually means missing information, not a bargain.")


def render_surebet_tab():
    st.info("A sure bet (arbitrage) is the only way to lock in profit: back every outcome at different "
            "bookmakers whose best odds are high enough. It is rare, short-lived, and bookmakers may limit accounts "
            "or void odds they call errors.")
    mk = st.radio("Market", ["1X2", "Two-way (Over/Under, BTTS...)"], horizontal=True, key="arb_mk")
    names = ["Home", "Draw", "Away"] if mk == "1X2" else ["Outcome A", "Outcome B"]
    nb = st.slider("Bookmakers", 2, 6, 3, key="arb_nb")
    odds = st.data_editor(pd.DataFrame(0.0, index=[f"Book {i + 1}" for i in range(nb)], columns=names),
                          use_container_width=True, key=f"arb_{mk}_{nb}")
    total = st.number_input("Total stake", 1.0, 1e9, 1000.0, key="arb_total")
    r = arbitrage(odds)
    if r is None:
        st.write("Enter odds (above 1.00) for every outcome.")
        return
    if r["profit_pct"] > 0:
        st.success(f"Sure bet found: guaranteed profit **{r['profit_pct']:.2f}%** "
                   f"(**{total / r['margin'] - total:,.2f}** on a {total:,.0f} stake).")
    else:
        st.warning(f"No sure bet: the best odds still carry a {-r['profit_pct']:.2f}% combined margin.")
    st.dataframe(pd.DataFrame({"Outcome": names, "Best odds": r["best"].values, "Bookmaker": r["book"].values,
                               "Stake": (total * r["w"]).round(2).values}), hide_index=True, use_container_width=True)
    st.caption("Re-check every price right before betting: if one moves, the guarantee disappears.")


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
