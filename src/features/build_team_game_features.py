from __future__ import annotations

import os
import re
from pathlib import Path
from pandas.errors import EmptyDataError

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser()
DUCKDB_PATH = Path(
    os.getenv("DUCKDB_PATH", str(PROJECT_ROOT / "data" / "march_madness.duckdb"))
).expanduser()
SEASON = int(os.getenv("SEASON", "2026"))
AS_OF_DATE = os.getenv("AS_OF_DATE", "").strip()

KAGGLE_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kaggle"

INJURIES_PATH = PROJECT_ROOT / "manual" / "injuries.csv"
COACH_ADJ_PATH = PROJECT_ROOT / "manual" / "coach_adjustments.csv"


def connect_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("create schema if not exists staging;")
    con.execute(
        """
        create table if not exists staging.team_season_snapshot_fact (
            season integer not null,
            as_of_date date not null,
            team_key varchar not null,
            games_played integer,
            wins integer,
            losses integer,
            net_rating double,
            adj_off double,
            adj_def double,
            tempo double,
            sos double,
            last_10_win_pct double,
            last_5_net_rating double,
            neutral_net_rating double,
            top_50_win_count integer,
            top_100_win_count integer,
            injury_adj double default 0,
            coach_adj double default 0,
            primary key (season, as_of_date, team_key)
        );
        """
    )
    return con


def canonical_team_key(season: int, team_name: str) -> str:
    s = str(team_name).lower().strip()
    s = s.replace("&", "and")
    s = s.replace("st.", "state")
    s = s.replace("saint", "st")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return f"{season}_{s}"


def get_effective_as_of_date_from_db(con: duckdb.DuckDBPyConnection) -> str | None:
    if AS_OF_DATE:
        return pd.to_datetime(AS_OF_DATE).date().isoformat()

    try:
        row = con.execute(
            """
            select cast(max(game_date) as date) as max_date
            from staging.game_dim
            where season = ?
            """,
            [SEASON],
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception:
        return None

    return None


def get_effective_as_of_date_fallback() -> str:
    if AS_OF_DATE:
        return pd.to_datetime(AS_OF_DATE).date().isoformat()
    return pd.Timestamp.today().date().isoformat()


def load_manual_adjustments(path: Path, kind: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        if kind == "injury":
            return pd.DataFrame(columns=["team_key", "injury_adj"])
        return pd.DataFrame(columns=["team_key", "coach_adj"])

    try:
        df = pd.read_csv(path).copy()
    except EmptyDataError:
        if kind == "injury":
            return pd.DataFrame(columns=["team_key", "injury_adj"])
        return pd.DataFrame(columns=["team_key", "coach_adj"])

    if "season" in df.columns:
        df = df[df["season"].astype(int) == SEASON].copy()

    if kind == "injury":
        if "team_key" not in df.columns or "injury_adj" not in df.columns:
            raise ValueError(f"{path.name} must include team_key and injury_adj")
        if "as_of_date" in df.columns:
            df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
        else:
            df["as_of_date"] = pd.Timestamp.today().date()
        if AS_OF_DATE:
            cutoff = pd.to_datetime(AS_OF_DATE).date()
            df = df[df["as_of_date"] <= cutoff].copy()
        df = (
            df.sort_values(["team_key", "as_of_date"])
            .groupby("team_key", as_index=False)
            .tail(1)[["team_key", "injury_adj"]]
        )
        df["injury_adj"] = df["injury_adj"].astype(float)
        return df

    if "team_key" not in df.columns or "coach_adj" not in df.columns:
        raise ValueError(f"{path.name} must include team_key and coach_adj")

    df = df[["team_key", "coach_adj"]].copy()
    df["coach_adj"] = df["coach_adj"].astype(float)
    return df.drop_duplicates(subset=["team_key"], keep="last")

def add_adjustments(out: pd.DataFrame) -> pd.DataFrame:
    injuries = load_manual_adjustments(INJURIES_PATH, kind="injury")
    coach = load_manual_adjustments(COACH_ADJ_PATH, kind="coach")

    out = out.merge(injuries, on="team_key", how="left")
    out = out.merge(coach, on="team_key", how="left")

    out["injury_adj"] = out["injury_adj"].fillna(0.0)
    out["coach_adj"] = out["coach_adj"].fillna(0.0)
    return out


def can_use_db_game_source(con: duckdb.DuckDBPyConnection) -> bool:
    try:
        row = con.execute(
            """
            select count(*)
            from staging.team_game_stats_fact t
            join staging.game_dim g
              on t.game_key = g.game_key
            where g.season = ?
            """,
            [SEASON],
        ).fetchone()
        return bool(row and row[0] and int(row[0]) > 0)
    except Exception:
        return False


def fetch_games_from_db(con: duckdb.DuckDBPyConnection, as_of_date: str) -> pd.DataFrame:
    df = con.execute(
        """
        select
            tgsf.game_key,
            gd.game_date,
            gd.neutral_site,
            tgsf.team_key,
            tgsf.opponent_team_key,
            tgsf.result,
            tgsf.points,
            tgsf.opp_points,
            tgsf.poss_est,
            tgsf.ortg,
            tgsf.drtg,
            tgsf.efg,
            tgsf.tov_rate,
            tgsf.orb_rate,
            tgsf.ftr,
            tgsf.opp_efg,
            tgsf.opp_tov_rate,
            tgsf.opp_orb_rate,
            tgsf.opp_ftr
        from staging.team_game_stats_fact tgsf
        join staging.game_dim gd
          on tgsf.game_key = gd.game_key
        where gd.season = ?
          and cast(gd.game_date as date) <= ?
        order by tgsf.team_key, gd.game_date
        """,
        [SEASON, as_of_date],
    ).df()

    if df.empty:
        raise RuntimeError("No current-season game rows found in staging.team_game_stats_fact.")

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["is_win"] = (df["result"].astype(str).str.upper() == "W").astype(int)
    df["net_game_rating"] = df["ortg"] - df["drtg"]
    df["neutral_site"] = df["neutral_site"].fillna(False).astype(bool)
    return df


def find_first_existing(candidates: list[str]) -> Path:
    for name in candidates:
        path = KAGGLE_RAW_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of these files in {KAGGLE_RAW_DIR}: {candidates}")


def maybe_find_first_existing(candidates: list[str]) -> Path | None:
    for name in candidates:
        path = KAGGLE_RAW_DIR / name
        if path.exists():
            return path
    return None


def load_kaggle_games() -> pd.DataFrame:
    teams = pd.read_csv(find_first_existing(["MTeams.csv"]))

    reg_detailed_path = maybe_find_first_existing(["MRegularSeasonDetailedResults.csv"])
    reg_compact_path = find_first_existing(["MRegularSeasonCompactResults.csv"])

    id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))

    if reg_detailed_path is not None:
        reg = pd.read_csv(reg_detailed_path)
        reg = reg[reg["Season"] == SEASON].copy()
        if reg.empty:
            raise RuntimeError(f"No {SEASON} rows found in {reg_detailed_path.name}")

        w = pd.DataFrame(
            {
                "team_name": reg["WTeamID"].map(id_to_name),
                "opp_name": reg["LTeamID"].map(id_to_name),
                "day_num": reg["DayNum"],
                "points": reg["WScore"],
                "opp_points": reg["LScore"],
                "is_win": 1,
                "neutral_site": (reg["WLoc"] == "N").astype(int),
                "fga": reg["WFGA"],
                "fgm": reg["WFGM"],
                "fg3a": reg["WFGA3"],
                "fg3m": reg["WFGM3"],
                "fta": reg["WFTA"],
                "ftm": reg["WFTM"],
                "orb": reg["WOR"],
                "drb": reg["WDR"],
                "tov": reg["WTO"],
                "opp_fga": reg["LFGA"],
                "opp_fgm": reg["LFGM"],
                "opp_fg3a": reg["LFGA3"],
                "opp_fg3m": reg["LFGM3"],
                "opp_fta": reg["LFTA"],
                "opp_ftm": reg["LFTM"],
                "opp_orb": reg["LOR"],
                "opp_drb": reg["LDR"],
                "opp_tov": reg["LTO"],
            }
        )

        l = pd.DataFrame(
            {
                "team_name": reg["LTeamID"].map(id_to_name),
                "opp_name": reg["WTeamID"].map(id_to_name),
                "day_num": reg["DayNum"],
                "points": reg["LScore"],
                "opp_points": reg["WScore"],
                "is_win": 0,
                "neutral_site": (reg["WLoc"] == "N").astype(int),
                "fga": reg["LFGA"],
                "fgm": reg["LFGM"],
                "fg3a": reg["LFGA3"],
                "fg3m": reg["LFGM3"],
                "fta": reg["LFTA"],
                "ftm": reg["LFTM"],
                "orb": reg["LOR"],
                "drb": reg["LDR"],
                "tov": reg["LTO"],
                "opp_fga": reg["WFGA"],
                "opp_fgm": reg["WFGM"],
                "opp_fg3a": reg["WFGA3"],
                "opp_fg3m": reg["WFGM3"],
                "opp_fta": reg["WFTA"],
                "opp_ftm": reg["WFTM"],
                "opp_orb": reg["WOR"],
                "opp_drb": reg["WDR"],
                "opp_tov": reg["WTO"],
            }
        )

        games = pd.concat([w, l], ignore_index=True)
        games["team_key"] = games["team_name"].map(lambda x: canonical_team_key(SEASON, x))
        games["opponent_team_key"] = games["opp_name"].map(lambda x: canonical_team_key(SEASON, x))

        games["poss_est"] = (
            games["fga"]
            - games["orb"]
            + games["tov"]
            + 0.475 * games["fta"]
        )
        opp_poss = (
            games["opp_fga"]
            - games["opp_orb"]
            + games["opp_tov"]
            + 0.475 * games["opp_fta"]
        )
        games["poss_est"] = (games["poss_est"] + opp_poss) / 2.0

        games["ortg"] = 100.0 * games["points"] / games["poss_est"].clip(lower=1e-9)
        games["drtg"] = 100.0 * games["opp_points"] / games["poss_est"].clip(lower=1e-9)
        games["net_game_rating"] = games["ortg"] - games["drtg"]

        games["efg"] = (games["fgm"] + 0.5 * games["fg3m"]) / games["fga"].clip(lower=1e-9)
        games["tov_rate"] = games["tov"] / games["poss_est"].clip(lower=1e-9)
        games["orb_rate"] = games["orb"] / (games["orb"] + games["opp_drb"]).clip(lower=1e-9)
        games["ftr"] = games["fta"] / games["fga"].clip(lower=1e-9)

        games["opp_efg"] = (games["opp_fgm"] + 0.5 * games["opp_fg3m"]) / games["opp_fga"].clip(lower=1e-9)
        games["opp_tov_rate"] = games["opp_tov"] / opp_poss.clip(lower=1e-9)
        games["opp_orb_rate"] = games["opp_orb"] / (games["opp_orb"] + games["drb"]).clip(lower=1e-9)
        games["opp_ftr"] = games["opp_fta"] / games["opp_fga"].clip(lower=1e-9)

        games["game_date"] = pd.NaT
        games["source"] = "kaggle_detailed"
        return games

    reg = pd.read_csv(reg_compact_path)
    reg = reg[reg["Season"] == SEASON].copy()
    if reg.empty:
        raise RuntimeError(f"No {SEASON} rows found in {reg_compact_path.name}")

    w = pd.DataFrame(
        {
            "team_name": reg["WTeamID"].map(id_to_name),
            "opp_name": reg["LTeamID"].map(id_to_name),
            "day_num": reg["DayNum"],
            "points": reg["WScore"],
            "opp_points": reg["LScore"],
            "is_win": 1,
            "neutral_site": (reg["WLoc"] == "N").astype(int),
        }
    )

    l = pd.DataFrame(
        {
            "team_name": reg["LTeamID"].map(id_to_name),
            "opp_name": reg["WTeamID"].map(id_to_name),
            "day_num": reg["DayNum"],
            "points": reg["LScore"],
            "opp_points": reg["WScore"],
            "is_win": 0,
            "neutral_site": (reg["WLoc"] == "N").astype(int),
        }
    )

    games = pd.concat([w, l], ignore_index=True)
    games["team_key"] = games["team_name"].map(lambda x: canonical_team_key(SEASON, x))
    games["opponent_team_key"] = games["opp_name"].map(lambda x: canonical_team_key(SEASON, x))

    games["poss_est"] = np.nan
    games["ortg"] = games["points"]
    games["drtg"] = games["opp_points"]
    games["net_game_rating"] = games["points"] - games["opp_points"]
    games["efg"] = np.nan
    games["tov_rate"] = np.nan
    games["orb_rate"] = np.nan
    games["ftr"] = np.nan
    games["opp_efg"] = np.nan
    games["opp_tov_rate"] = np.nan
    games["opp_orb_rate"] = np.nan
    games["opp_ftr"] = np.nan
    games["game_date"] = pd.NaT
    games["source"] = "kaggle_compact"
    return games


def build_base_team_strength(games: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        games.groupby("team_key", as_index=False)
        .agg(
            games_played=("is_win", "count"),
            wins=("is_win", "sum"),
            losses=("is_win", lambda s: int(len(s) - s.sum())),
            points=("points", "sum"),
            opp_points=("opp_points", "sum"),
            poss_est=("poss_est", "sum"),
            tempo=("poss_est", "mean"),
            efg=("efg", "mean"),
            tov_rate=("tov_rate", "mean"),
            orb_rate=("orb_rate", "mean"),
            ftr=("ftr", "mean"),
            opp_efg=("opp_efg", "mean"),
            opp_tov_rate=("opp_tov_rate", "mean"),
            opp_orb_rate=("opp_orb_rate", "mean"),
            opp_ftr=("opp_ftr", "mean"),
        )
    )

    use_efficiency = grouped["poss_est"].notna() & (grouped["poss_est"] > 0)

    grouped["adj_off"] = np.where(
        use_efficiency,
        100.0 * grouped["points"] / grouped["poss_est"].clip(lower=1e-9),
        grouped["points"] / grouped["games_played"].clip(lower=1),
    )
    grouped["adj_def"] = np.where(
        use_efficiency,
        100.0 * grouped["opp_points"] / grouped["poss_est"].clip(lower=1e-9),
        grouped["opp_points"] / grouped["games_played"].clip(lower=1),
    )
    grouped["net_rating"] = grouped["adj_off"] - grouped["adj_def"]

    grouped["tempo"] = grouped["tempo"].fillna(0.0)
    return grouped


def add_recent_form(games: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["team_key", "game_date"] if games["game_date"].notna().any() else ["team_key", "day_num"]

    last10 = (
        games.sort_values(sort_cols)
        .groupby("team_key")
        .tail(10)
        .groupby("team_key", as_index=False)
        .agg(last_10_win_pct=("is_win", "mean"))
    )

    last5 = (
        games.sort_values(sort_cols)
        .groupby("team_key")
        .tail(5)
        .groupby("team_key", as_index=False)
        .agg(last_5_net_rating=("net_game_rating", "mean"))
    )

    neutral = (
        games[games["neutral_site"].astype(bool)]
        .groupby("team_key", as_index=False)
        .agg(neutral_net_rating=("net_game_rating", "mean"))
    )

    out = out.merge(last10, on="team_key", how="left")
    out = out.merge(last5, on="team_key", how="left")
    out = out.merge(neutral, on="team_key", how="left")

    out["last_10_win_pct"] = out["last_10_win_pct"].fillna(0.0)
    out["last_5_net_rating"] = out["last_5_net_rating"].fillna(out["net_rating"])
    out["neutral_net_rating"] = out["neutral_net_rating"].fillna(out["net_rating"])
    return out


def add_sos_and_quality_wins(games: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    team_strength = out[["team_key", "net_rating"]].rename(
        columns={"team_key": "opponent_team_key", "net_rating": "opp_strength"}
    )

    enriched = games.merge(team_strength, on="opponent_team_key", how="left")
    enriched["opp_strength"] = enriched["opp_strength"].fillna(0.0)

    sos = (
        enriched.groupby("team_key", as_index=False)
        .agg(sos=("opp_strength", "mean"))
    )

    opp_ranks = (
        out[["team_key", "net_rating"]]
        .sort_values("net_rating", ascending=False)
        .reset_index(drop=True)
    )
    opp_ranks["rank_by_net"] = opp_ranks.index + 1
    opp_ranks = opp_ranks.rename(
        columns={"team_key": "opponent_team_key"}
    )[["opponent_team_key", "rank_by_net"]]

    enriched = enriched.merge(opp_ranks, on="opponent_team_key", how="left")
    enriched["top_50_win"] = ((enriched["is_win"] == 1) & (enriched["rank_by_net"] <= 50)).astype(int)
    enriched["top_100_win"] = ((enriched["is_win"] == 1) & (enriched["rank_by_net"] <= 100)).astype(int)

    quality = (
        enriched.groupby("team_key", as_index=False)
        .agg(
            top_50_win_count=("top_50_win", "sum"),
            top_100_win_count=("top_100_win", "sum"),
        )
    )

    out = out.merge(sos, on="team_key", how="left")
    out = out.merge(quality, on="team_key", how="left")

    out["sos"] = out["sos"].fillna(0.0)
    out["top_50_win_count"] = out["top_50_win_count"].fillna(0).astype(int)
    out["top_100_win_count"] = out["top_100_win_count"].fillna(0).astype(int)
    return out


def persist_snapshot(
    con: duckdb.DuckDBPyConnection,
    out: pd.DataFrame,
    as_of_date: str,
) -> int:
    out = out.copy()
    out["season"] = SEASON
    out["as_of_date"] = pd.to_datetime(as_of_date).date()

    cols = [
        "season",
        "as_of_date",
        "team_key",
        "games_played",
        "wins",
        "losses",
        "net_rating",
        "adj_off",
        "adj_def",
        "tempo",
        "sos",
        "last_10_win_pct",
        "last_5_net_rating",
        "neutral_net_rating",
        "top_50_win_count",
        "top_100_win_count",
        "injury_adj",
        "coach_adj",
    ]

    con.execute(
        "delete from staging.team_season_snapshot_fact where season = ? and as_of_date = ?",
        [SEASON, as_of_date],
    )

    payload = list(out[cols].itertuples(index=False, name=None))
    con.executemany(
        """
        insert into staging.team_season_snapshot_fact
        (season, as_of_date, team_key, games_played, wins, losses, net_rating,
         adj_off, adj_def, tempo, sos, last_10_win_pct, last_5_net_rating,
         neutral_net_rating, top_50_win_count, top_100_win_count, injury_adj, coach_adj)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def main() -> int:
    con = connect_db()

    db_as_of_date = get_effective_as_of_date_from_db(con)

    if can_use_db_game_source(con):
        as_of_date = db_as_of_date or get_effective_as_of_date_fallback()
        games = fetch_games_from_db(con, as_of_date=as_of_date)
        source = "database"
    else:
        as_of_date = get_effective_as_of_date_fallback()
        games = load_kaggle_games()
        source = str(games["source"].iloc[0])

    out = build_base_team_strength(games)
    out = add_recent_form(games, out)
    out = add_sos_and_quality_wins(games, out)
    out = add_adjustments(out)

    n = persist_snapshot(con, out, as_of_date=as_of_date)

    print(f"Built team season snapshot for {as_of_date}")
    print(f"Source: {source}")
    print(f"Inserted rows: {n}")

    preview = out.sort_values("net_rating", ascending=False)[
        ["team_key", "wins", "losses", "net_rating", "adj_off", "adj_def", "sos"]
    ].head(15)
    print("\nTop 15 by net_rating:")
    print(preview.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())