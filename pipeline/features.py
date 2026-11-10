"""
Feature engineering for next-game stat predictions.

Every feature for a game uses only information available before tip-off: the player's
earlier games, last season's averages, rest, home or away, and how the opponent has
defended so far that season. That's what makes the backtest honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATS = {
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "fg3": "3_pt_made",
}

# Exhibition opponents (All-Star teams etc.) carry an asterisk in the source data
EXHIBITION = r"\*"


def load_games(path: str) -> pd.DataFrame:
    g = pd.read_csv(path)
    g = g[~g["opponent"].str.contains(EXHIBITION, regex=True)].copy()
    g = g[g["minutes"] > 0]
    # Dates are "month/day"; Sep–Dec belong to the calendar year before the season label
    md = g["date"].str.split("/", expand=True).astype(int)
    year = np.where(md[0] >= 9, g["season_year"] - 1, g["season_year"])
    g["game_date"] = pd.to_datetime({"year": year, "month": md[0], "day": md[1]})
    g = g.rename(columns={v: k for k, v in STATS.items()})
    g["home"] = (g["stadium"] == "home").astype(int)
    g["fga"] = g["field_goals_attempted"]
    g["fg3a"] = g["3_pt_attempted"]
    g["fta"] = g["free_throws_attempted"]
    keep = ["id", "season_year", "game_date", "opponent", "home", "minutes", "fga", "fg3a", "fta", *STATS]
    return g[keep].sort_values(["id", "game_date"]).reset_index(drop=True)


def add_player_form(g: pd.DataFrame) -> pd.DataFrame:
    """Rolling and season-to-date averages from the player's earlier games (shifted by one)."""
    g = g.copy()
    by_player = g.groupby("id", sort=False)
    by_season = g.groupby(["id", "season_year"], sort=False)
    g["game_num"] = by_season.cumcount() + 1
    rest = by_player["game_date"].diff().dt.days
    g["rest_days"] = rest.clip(upper=7).fillna(7)
    g["back_to_back"] = (rest == 1).astype(int)
    for col in ["minutes", "fga", "fg3a", "fta", *STATS]:
        prev = by_player[col].shift(1)
        g[f"{col}_l5"] = prev.groupby(g["id"]).transform(lambda s: s.rolling(5, min_periods=1).mean())
        g[f"{col}_l10"] = prev.groupby(g["id"]).transform(lambda s: s.rolling(10, min_periods=1).mean())
        season_prev = by_season[col].shift(1)
        g[f"{col}_season"] = season_prev.groupby([g["id"], g["season_year"]]).transform(
            lambda s: s.expanding().mean()
        )
    for col in STATS:
        prev = by_player[col].shift(1)
        g[f"{col}_ewm"] = prev.groupby(g["id"]).transform(lambda s: s.ewm(halflife=6, min_periods=1).mean())
    return g


def add_last_season(g: pd.DataFrame, averages_path: str, info_path: str) -> pd.DataFrame:
    """Last season's per-game averages, plus age and experience."""
    avg = pd.read_csv(averages_path)
    avg = avg.groupby(["id", "season_year"]).tail(1)  # traded players: keep final team row
    cols = {
        "season_average_points": "pts_last_season",
        "season_average_rebounds": "reb_last_season",
        "season_average_assists": "ast_last_season",
        "season_average_3_pt_made": "fg3_last_season",
        "season_average_minutes": "minutes_last_season",
    }
    prev = avg[["id", "season_year", *cols]].rename(columns=cols)
    prev["season_year"] = prev["season_year"] + 1  # last season's numbers describe this season's games
    first = avg.groupby("id")["season_year"].min().rename("first_season")
    info = pd.read_csv(info_path)[["id", "birth_year"]]
    g = g.merge(prev, on=["id", "season_year"], how="left").merge(first, on="id", how="left").merge(info, on="id", how="left")
    g["experience"] = (g["season_year"] - g["first_season"]).clip(lower=0)
    g["age"] = g["season_year"] - g["birth_year"]
    return g.drop(columns=["first_season", "birth_year"])


def opponent_table(g: pd.DataFrame, prior_games: int = 40) -> pd.DataFrame:
    """
    How much of each stat opponents allowed per player-game so far that season, as a ratio to
    the league average. Built date by date, so a game never sees its own result, and shrunk
    toward 1.0 early in the season when a team has few games behind it.
    """
    stats = list(STATS)
    daily = g.groupby(["season_year", "opponent", "game_date"])[stats].agg(["sum", "count"])
    daily.columns = [f"{s}_{a}" for s, a in daily.columns]
    daily = daily.reset_index().sort_values(["season_year", "opponent", "game_date"])

    league = g.groupby(["season_year", "game_date"])[stats].agg(["sum", "count"])
    league.columns = [f"{s}_{a}" for s, a in league.columns]
    league = league.reset_index().sort_values(["season_year", "game_date"])
    for s in stats:
        lg = league.groupby("season_year")
        before_sum = lg[f"{s}_sum"].cumsum() - league[f"{s}_sum"]
        before_n = lg[f"{s}_count"].cumsum() - league[f"{s}_count"]
        league[f"{s}_league"] = before_sum / before_n.replace(0, np.nan)

    out = daily.merge(league[["season_year", "game_date", *[f"{s}_league" for s in stats]]], on=["season_year", "game_date"], how="left")
    grp = out.groupby(["season_year", "opponent"])
    for s in stats:
        before_sum = grp[f"{s}_sum"].cumsum() - out[f"{s}_sum"]
        before_n = grp[f"{s}_count"].cumsum() - out[f"{s}_count"]
        lg_mean = out[f"{s}_league"]
        shrunk = (before_sum + prior_games * lg_mean) / (before_n + prior_games)
        out[f"opp_{s}_factor"] = (shrunk / lg_mean).fillna(1.0)
    return out[["season_year", "opponent", "game_date", *[f"opp_{s}_factor" for s in stats]]]


def add_opponent(g: pd.DataFrame) -> pd.DataFrame:
    return g.merge(opponent_table(g), on=["season_year", "opponent", "game_date"], how="left")


def feature_columns(stat: str) -> list[str]:
    """Inputs for one stat's model."""
    return [
        f"{stat}_l5", f"{stat}_l10", f"{stat}_season", f"{stat}_ewm", f"{stat}_last_season",
        "minutes_l5", "minutes_l10", "minutes_season", "minutes_last_season",
        "fga_l10", "fg3a_l10", "fta_l10",
        f"opp_{stat}_factor", "home", "rest_days", "back_to_back", "game_num", "age", "experience",
    ]


def build(games_path: str, averages_path: str, info_path: str) -> pd.DataFrame:
    g = load_games(games_path)
    g = add_player_form(g)
    g = add_last_season(g, averages_path, info_path)
    return add_opponent(g)
