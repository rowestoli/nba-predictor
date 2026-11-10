"""
Train next-game models, backtest them honestly, and export everything the web app needs.

    python pipeline/train.py

Splits (by season, so no game from the future leaks into training):
    train      2015–2023 seasons
    validate   2023–24 season    (used to calibrate over/under probabilities)
    test       2024–25 season    (games through the data snapshot, never seen before)

Outputs in public/data/: players.json, players/<id>.json, model.json
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

import features as F

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "public" / "data"
GAMES, AVERAGES, INFO = RAW / "players_gamelogs.csv", RAW / "players_season_averages.csv", RAW / "players_info.csv"
CURRENT = 2025
QUANTILES = np.linspace(0.02, 0.98, 49)
BINS = 8  # prediction buckets for the residual distributions

PARAMS = dict(loss="poisson", max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
              min_samples_leaf=80, l2_regularization=1.0, random_state=0)

TEAMS = {
    "ATL": "Atlanta Hawks", "BKN": "Brooklyn Nets", "BOS": "Boston Celtics", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GS": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NO": "New Orleans Pelicans", "NY": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SA": "San Antonio Spurs", "SAC": "Sacramento Kings", "TOR": "Toronto Raptors",
    "UTAH": "Utah Jazz", "WSH": "Washington Wizards",
}


def baseline(df: pd.DataFrame, stat: str, fallback: float) -> pd.Series:
    """What a fan would guess: season average so far, else last season's, else recent form."""
    return df[f"{stat}_season"].fillna(df[f"{stat}_last_season"]).fillna(df[f"{stat}_l10"]).fillna(fallback)


def residual_model(pred: np.ndarray, actual: np.ndarray) -> dict:
    """Empirical residual quantiles in buckets of the prediction, used for over/under odds."""
    edges = np.quantile(pred, np.linspace(0, 1, BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, BINS - 1)
    quantiles = []
    for b in range(BINS):
        r = actual[idx == b] - pred[idx == b]
        quantiles.append(np.round(np.quantile(r, QUANTILES), 2).tolist())
    return {"edges": [None if not np.isfinite(e) else round(float(e), 3) for e in edges], "quantiles": quantiles}


def prob_over(pred: np.ndarray, line: np.ndarray, rm: dict) -> np.ndarray:
    """P(actual > line) from the residual distribution of the prediction's bucket."""
    edges = np.array([-np.inf if e is None else e for e in rm["edges"]])
    edges[-1] = np.inf
    idx = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, BINS - 1)
    q = np.array(rm["quantiles"])[idx]  # (n, 49)
    return ((pred[:, None] + q) > line[:, None]).mean(axis=1)


def calibration(p: np.ndarray, hit: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.sum() >= 30:
            out.append({"predicted": round(float(p[m].mean()), 3), "actual": round(float(hit[m].mean()), 3), "n": int(m.sum())})
    return out


def next_game_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """One placeholder game per current player, two days after their latest game."""
    cur = raw[raw["season_year"] == CURRENT]
    last = cur.groupby("id").tail(1)
    rows = last[["id", "season_year", "game_date"]].copy()
    rows["game_date"] = rows["game_date"] + timedelta(days=2)
    rows["opponent"] = "TBD"
    rows["home"] = 0
    rows["is_future"] = True
    return rows


def main() -> None:
    raw = F.load_games(GAMES)
    future = next_game_rows(raw)
    allg = pd.concat([raw.assign(is_future=False), future], ignore_index=True).sort_values(["id", "game_date"])
    allg = F.add_player_form(allg.reset_index(drop=True))
    allg = F.add_last_season(allg, AVERAGES, INFO)
    opp = F.opponent_table(raw)
    allg = allg.merge(opp, on=["season_year", "opponent", "game_date"], how="left")
    games = allg[~allg["is_future"]].copy()
    upcoming = allg[allg["is_future"]].copy()

    train = games[games["season_year"] <= CURRENT - 2]
    val = games[games["season_year"] == CURRENT - 1]
    test = games[games["season_year"] == CURRENT]
    print(f"train {len(train):,}  validate {len(val):,}  test {len(test):,}")

    # Each opponent's defense as of the latest date in the data (shrunk toward average)
    latest_opp = opp[opp["season_year"] == CURRENT].sort_values("game_date").groupby("opponent").tail(1).set_index("opponent")
    teams = [t for t in TEAMS if t in latest_opp.index]

    metrics, models, residuals, importance = {}, {}, {}, {}
    for stat in F.STATS:
        cols = F.feature_columns(stat)
        # 1. Backtest model: trained on old seasons only
        m = HistGradientBoostingRegressor(**PARAMS).fit(train[cols], train[stat])
        pv, pt = m.predict(val[cols]), m.predict(test[cols])
        rm = residual_model(pv, val[stat].to_numpy())
        fallback = train[stat].mean()
        bt = baseline(test, stat, fallback)
        mae_model, mae_base = mean_absolute_error(test[stat], pt), mean_absolute_error(test[stat], bt)

        # Over/under check against a line at the player's season average (rounded to .5).
        # Stat lines are right-skewed, so "always take the under" is the naive strategy to beat.
        line = np.floor(bt.to_numpy()) + 0.5
        p = prob_over(pt, line, rm)
        hit = (test[stat].to_numpy() > line).astype(float)
        recent = test[f"{stat}_l10"].fillna(bt).to_numpy()
        metrics[stat] = {
            "label": {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3": "3-pointers made"}[stat],
            "mae_model": round(float(mae_model), 3),
            "mae_baseline": round(float(mae_base), 3),
            "improvement": round(float(1 - mae_model / mae_base), 4),
            "pick_accuracy": round(float(((p > 0.5) == (hit == 1)).mean()), 4),
            "always_under": round(float((hit == 0).mean()), 4),
            "recent_form_rule": round(float(((recent > line) == (hit == 1)).mean()), 4),
            "brier": round(float(np.mean((p - hit) ** 2)), 4),
            "brier_base_rate": round(float(np.mean((hit.mean() - hit) ** 2)), 4),
            "calibration": calibration(p, hit),
        }
        residuals[stat] = rm
        print(f"{stat}: MAE {mae_model:.3f} vs baseline {mae_base:.3f} ({metrics[stat]['improvement']:.1%} better); "
              f"over/under picks {metrics[stat]['pick_accuracy']:.1%} vs always-under {metrics[stat]['always_under']:.1%}")

        if stat == "pts":
            sample = val.sample(6000, random_state=0)
            pi = permutation_importance(m, sample[cols], sample[stat], scoring="neg_mean_absolute_error", n_repeats=4, random_state=0)
            ranked = sorted(zip(cols, pi.importances_mean), key=lambda x: -x[1])[:8]
            importance = [{"feature": f, "mae_increase": round(float(v), 3)} for f, v in ranked]

        # 2. Production model: same settings, trained on everything up to the snapshot
        models[stat] = HistGradientBoostingRegressor(**PARAMS).fit(games[cols], games[stat])

    # ---- predictions for each player's next game, for every opponent, home and away ----
    info = pd.read_csv(INFO).set_index("id")
    avgs = pd.read_csv(AVERAGES)
    cur_team = avgs[avgs["season_year"] == CURRENT].groupby("id").tail(1).set_index("id")
    (OUT / "players").mkdir(parents=True, exist_ok=True)
    index = []
    for _, row in upcoming.iterrows():
        pid = int(row["id"])
        hist = games[(games["id"] == pid) & (games["season_year"] == CURRENT)]
        if len(hist) < 3 or pid not in info.index:
            continue
        team = cur_team.loc[pid, "season_team"] if pid in cur_team.index else None
        scenarios = []
        for t in teams:
            if t == team:
                continue
            for home in (1, 0):
                r = row.copy()
                r["home"] = home
                for stat in F.STATS:
                    r[f"opp_{stat}_factor"] = latest_opp.loc[t, f"opp_{stat}_factor"]
                scenarios.append((t, home, r))
        frame = pd.DataFrame([s[2] for s in scenarios])
        preds = {stat: np.round(models[stat].predict(frame[F.feature_columns(stat)]), 2).tolist() for stat in F.STATS}
        recent = hist.tail(15)
        player = {
            "id": pid,
            "name": info.loc[pid, "name"],
            "team": team,
            "position": cur_team.loc[pid, "position"] if pid in cur_team.index else None,
            "season": {s: round(float(hist[s].mean()), 1) for s in [*F.STATS, "minutes"]},
            "games": int(len(hist)),
            "last_season": {s: (None if pd.isna(row[f"{s}_last_season"]) else round(float(row[f"{s}_last_season"]), 1)) for s in F.STATS},
            "recent": [
                {"date": d.strftime("%b %-d"), "opp": o, "home": int(h), "min": int(mi), **{s: int(v[s]) for s in F.STATS}}
                for d, o, h, mi, v in zip(recent["game_date"], recent["opponent"], recent["home"], recent["minutes"], recent[list(F.STATS)].to_dict("records"))
            ],
            "scenarios": [{"opp": t, "home": h} for t, h, _ in scenarios],
            "predictions": preds,
        }
        write(OUT / "players" / f"{pid}.json", player)
        index.append({"id": pid, "name": player["name"], "team": team, "position": player["position"],
                      "pts": player["season"]["pts"], "games": player["games"]})

    index.sort(key=lambda p: -p["pts"])
    write(OUT / "players.json", index)
    snapshot = games["game_date"].max()
    write(OUT / "model.json", {
        "snapshot": snapshot.strftime("%B %-d, %Y"),
        "season": f"{CURRENT - 1}–{str(CURRENT)[2:]}",
        "split": {"train": int(len(train)), "validate": int(len(val)), "test": int(len(test))},
        "metrics": metrics,
        "residuals": residuals,
        "importance": importance,
        "teams": {t: TEAMS[t] for t in teams},
        "defense": {t: {s: round(float(latest_opp.loc[t, f"opp_{s}_factor"]), 3) for s in F.STATS} for t in teams},
    })
    print(f"Exported {len(index)} players; data through {snapshot:%b %d, %Y}")


def write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
