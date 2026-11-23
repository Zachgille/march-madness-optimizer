from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.ids import canonical_team_key, normalize_team_name

load_dotenv()

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser()
DUCKDB_PATH = Path(
    os.getenv("DUCKDB_PATH", str(PROJECT_ROOT / "data" / "march_madness.duckdb"))
).expanduser()
SEASON = int(os.getenv("SEASON", "2026"))

CBD_API_KEY = os.getenv("CBD_API_KEY", "").strip()
CBD_INPUT_CSV = os.getenv("CBD_INPUT_CSV", "").strip()
CBD_EXPORTER_URL = os.getenv("CBD_EXPORTER_URL", "").strip()
CBD_EXPORTER_PARAMS_JSON = os.getenv("CBD_EXPORTER_PARAMS_JSON", "").strip()

RAW_OUT_DIR = PROJECT_ROOT / "data" / "raw" / "collegebasketballdata"


def connect_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("create schema if not exists raw;")
    con.execute("create schema if not exists staging;")

    con.execute(
        """
        create table if not exists raw.raw_source_snapshot (
            snapshot_id varchar primary key,
            source_name varchar not null,
            fetched_at timestamp not null,
            season integer not null,
            file_path varchar not null,
            checksum varchar,
            notes varchar
        );
        """
    )

    con.execute(
        """
        create table if not exists staging.team_dim (
            team_key varchar primary key,
            season integer not null,
            team_name varchar not null,
            short_name varchar,
            conference varchar,
            division varchar default 'D1',
            kaggle_team_id integer,
            cbd_team_id varchar,
            ncaa_team_id varchar,
            yahoo_team_name varchar,
            espn_team_name varchar,
            is_active boolean default true
        );
        """
    )

    con.execute(
        """
        create table if not exists staging.team_name_crosswalk (
            alias_name varchar not null,
            normalized_alias varchar not null,
            source_name varchar not null,
            season integer,
            team_key varchar not null,
            primary key (alias_name, source_name, season)
        );
        """
    )

    con.execute(
        """
        create table if not exists staging.game_dim (
            game_key varchar primary key,
            season integer not null,
            source_game_id varchar,
            game_date timestamp,
            neutral_site boolean,
            venue varchar,
            city varchar,
            state varchar,
            is_postseason boolean,
            tournament_name varchar,
            round_name varchar,
            round_num integer,
            region varchar,
            slot varchar,
            team1_key varchar,
            team2_key varchar,
            team1_score integer,
            team2_score integer,
            winner_team_key varchar,
            loser_team_key varchar
        );
        """
    )

    con.execute(
        """
        create table if not exists staging.team_game_stats_fact (
            game_key varchar not null,
            team_key varchar not null,
            opponent_team_key varchar not null,
            is_home boolean,
            is_away boolean,
            is_neutral boolean,
            result varchar,
            points integer,
            opp_points integer,
            fgm integer,
            fga integer,
            fg3m integer,
            fg3a integer,
            ftm integer,
            fta integer,
            orb integer,
            drb integer,
            trb integer,
            ast integer,
            stl integer,
            blk integer,
            tov integer,
            pf integer,
            poss_est double,
            ortg double,
            drtg double,
            efg double,
            tov_rate double,
            orb_rate double,
            ftr double,
            opp_efg double,
            opp_tov_rate double,
            opp_orb_rate double,
            opp_ftr double,
            primary key (game_key, team_key)
        );
        """
    )
    return con


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = []
    for c in out.columns:
        c2 = str(c).strip().lower()
        c2 = c2.replace("%", "pct")
        c2 = c2.replace("/", "_")
        c2 = c2.replace("-", "_")
        c2 = c2.replace(" ", "_")
        while "__" in c2:
            c2 = c2.replace("__", "_")
        cols.append(c2)
    out.columns = cols
    return out


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def latest_csv_in_raw_dir() -> Path | None:
    if not RAW_OUT_DIR.exists():
        return None
    csvs = sorted(RAW_OUT_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def fetch_exporter_csv() -> Path:
    if not CBD_EXPORTER_URL:
        raise RuntimeError(
            "CBD_EXPORTER_URL is not set. Either set CBD_INPUT_CSV or place a CSV in data/raw/collegebasketballdata."
        )

    params: dict[str, Any] = {}
    if CBD_EXPORTER_PARAMS_JSON:
        params = json.loads(CBD_EXPORTER_PARAMS_JSON)

    headers = {}
    if CBD_API_KEY:
        headers["Authorization"] = f"Bearer {CBD_API_KEY}"

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)

    resp = requests.get(CBD_EXPORTER_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()

    ts = utc_now_str()
    out_path = RAW_OUT_DIR / f"cbd_export_{SEASON}_{ts}.csv"
    out_path.write_bytes(resp.content)
    return out_path


def locate_or_fetch_source() -> Path:
    if CBD_INPUT_CSV:
        path = Path(CBD_INPUT_CSV).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"CBD_INPUT_CSV does not exist: {path}")
        return path

    latest = latest_csv_in_raw_dir()
    if latest is not None:
        return latest

    return fetch_exporter_csv()


def record_snapshot(con: duckdb.DuckDBPyConnection, file_path: Path, note: str) -> None:
    payload = file_path.read_bytes()
    snapshot_id = f"collegebasketballdata_{utc_now_str()}_{file_path.name}"
    con.execute(
        """
        insert or replace into raw.raw_source_snapshot
        (snapshot_id, source_name, fetched_at, season, file_path, checksum, notes)
        values (?, 'collegebasketballdata', current_timestamp, ?, ?, ?, ?)
        """,
        [snapshot_id, SEASON, str(file_path), sha256_bytes(payload), note],
    )


def load_source_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    return df


def parse_bool(x: object) -> bool:
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_game_key(
    season: int,
    game_date: str,
    team1: str,
    team2: str,
    source_game_id: str | None = None,
) -> str:
    a, b = sorted([normalize_team_name(team1), normalize_team_name(team2)])
    raw = f"{season}|{game_date}|{a}|{b}|{source_game_id or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def infer_postseason(row: pd.Series) -> bool:
    postseason_col = row.get("is_postseason")
    if postseason_col is not None and not pd.isna(postseason_col):
        return parse_bool(postseason_col)

    round_name = str(row.get("round_name", "") or "").lower()
    tournament_name = str(row.get("tournament_name", "") or "").lower()
    hay = f"{round_name} {tournament_name}"
    keywords = [
        "ncaa",
        "march madness",
        "sweet 16",
        "elite eight",
        "final four",
        "round of 64",
        "round of 32",
        "conference tournament",
        "sec tournament",
        "big ten tournament",
        "acc tournament",
        "big 12 tournament",
    ]
    return any(k in hay for k in keywords)


def ensure_team_records(
    con: duckdb.DuckDBPyConnection,
    season_df: pd.DataFrame,
) -> tuple[dict[str, str], int, int]:
    existing_aliases = con.execute(
        """
        select normalized_alias, team_key
        from staging.team_name_crosswalk
        where season = ?
        """,
        [SEASON],
    ).df()

    alias_map: dict[str, str] = {}
    for _, row in existing_aliases.iterrows():
        alias_map.setdefault(str(row["normalized_alias"]), str(row["team_key"]))

    existing_team_keys = set(
        con.execute(
            "select team_key from staging.team_dim where season = ?",
            [SEASON],
        ).df()["team_key"].tolist()
    )

    names = sorted(
        set(season_df["team_name"].dropna().astype(str).str.strip().tolist())
        | set(season_df["opponent_name"].dropna().astype(str).str.strip().tolist())
    )

    new_team_rows: list[tuple] = []
    new_alias_rows: list[tuple] = []

    for name in names:
        norm = normalize_team_name(name)
        team_key = alias_map.get(norm)
        if not team_key:
            team_key = canonical_team_key(SEASON, name)

        if team_key not in existing_team_keys:
            new_team_rows.append(
                (
                    team_key,
                    SEASON,
                    name,
                    name,
                    None,
                    "D1",
                    None,
                    None,
                    None,
                    None,
                    None,
                    True,
                )
            )
            existing_team_keys.add(team_key)

        if norm not in alias_map:
            new_alias_rows.append((name, norm, "collegebasketballdata", SEASON, team_key))
            alias_map[norm] = team_key

    if new_team_rows:
        con.executemany(
            """
            insert or replace into staging.team_dim
            (team_key, season, team_name, short_name, conference, division,
             kaggle_team_id, cbd_team_id, ncaa_team_id, yahoo_team_name, espn_team_name, is_active)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            new_team_rows,
        )

    if new_alias_rows:
        con.executemany(
            """
            insert or replace into staging.team_name_crosswalk
            (alias_name, normalized_alias, source_name, season, team_key)
            values (?, ?, ?, ?, ?)
            """,
            new_alias_rows,
        )

    return alias_map, len(new_team_rows), len(new_alias_rows)


def standardize_long_format(df: pd.DataFrame) -> pd.DataFrame:
    team_col = find_col(df, ["team_name", "team", "school", "team"])
    opp_col = find_col(df, ["opponent_name", "opponent", "opp_name", "opp"])
    date_col = find_col(df, ["game_date", "date", "start_date", "start_time"])
    pts_col = find_col(df, ["points", "team_points", "score", "pts"])
    opp_pts_col = find_col(df, ["opp_points", "opponent_points", "opp_score", "opponent_score"])

    if not all([team_col, opp_col, date_col, pts_col, opp_pts_col]):
        raise ValueError(
            "Long-format CBD file missing required columns. "
            "Need team/opponent/date/points/opp_points equivalents."
        )

    source_game_id_col = find_col(df, ["game_id", "event_id", "id", "matchup_id"])
    neutral_col = find_col(df, ["is_neutral", "neutral_site", "neutral"])
    home_col = find_col(df, ["is_home", "home"])
    away_col = find_col(df, ["is_away", "away"])
    venue_col = find_col(df, ["venue"])
    city_col = find_col(df, ["city"])
    state_col = find_col(df, ["state"])
    tournament_col = find_col(df, ["tournament_name", "tournament"])
    round_col = find_col(df, ["round_name", "round"])
    postseason_col = find_col(df, ["is_postseason", "postseason"])

    mapping = {
        "team_name": team_col,
        "opponent_name": opp_col,
        "game_date": date_col,
        "points": pts_col,
        "opp_points": opp_pts_col,
    }

    out = pd.DataFrame(
        {
            "source_game_id": df[source_game_id_col].astype(str) if source_game_id_col else None,
            "game_date": pd.to_datetime(df[date_col], errors="coerce"),
            "neutral_site": df[neutral_col].map(parse_bool) if neutral_col else False,
            "is_home": df[home_col].map(parse_bool) if home_col else False,
            "is_away": df[away_col].map(parse_bool) if away_col else False,
            "venue": df[venue_col] if venue_col else None,
            "city": df[city_col] if city_col else None,
            "state": df[state_col] if state_col else None,
            "tournament_name": df[tournament_col] if tournament_col else None,
            "round_name": df[round_col] if round_col else None,
            "is_postseason": df[postseason_col].map(parse_bool) if postseason_col else None,
            "team_name": df[team_col].astype(str).str.strip(),
            "opponent_name": df[opp_col].astype(str).str.strip(),
            "points": coerce_numeric(df[pts_col]),
            "opp_points": coerce_numeric(df[opp_pts_col]),
        }
    )

    # Optional box-score stats
    stat_aliases = {
        "fgm": ["fgm", "field_goals_made", "field_goal_made"],
        "fga": ["fga", "field_goals_attempted", "field_goal_attempted"],
        "fg3m": ["fg3m", "three_pointers_made", "3pm", "three_point_made"],
        "fg3a": ["fg3a", "three_pointers_attempted", "3pa", "three_point_attempted"],
        "ftm": ["ftm", "free_throws_made"],
        "fta": ["fta", "free_throws_attempted"],
        "orb": ["orb", "offensive_rebounds"],
        "drb": ["drb", "defensive_rebounds"],
        "trb": ["trb", "rebounds", "total_rebounds"],
        "ast": ["ast", "assists"],
        "stl": ["stl", "steals"],
        "blk": ["blk", "blocks"],
        "tov": ["tov", "turnovers"],
        "pf": ["pf", "personal_fouls", "fouls"],
    }

    for target, aliases in stat_aliases.items():
        col = find_col(df, aliases)
        out[target] = coerce_numeric(df[col]) if col else None

    out["neutral_site"] = out["neutral_site"].fillna(False)
    out["is_home"] = out["is_home"].fillna(False)
    out["is_away"] = out["is_away"].fillna(False)

    # If location flags missing, infer neutral if neither home nor away
    neither = (~out["is_home"]) & (~out["is_away"])
    out.loc[neither, "neutral_site"] = out.loc[neither, "neutral_site"] | True

    out["game_date"] = out["game_date"].dt.tz_localize(None)
    return out


def standardize_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    home_team_col = find_col(df, ["home_team", "home_team_name"])
    away_team_col = find_col(df, ["away_team", "away_team_name"])
    date_col = find_col(df, ["game_date", "date", "start_date", "start_time"])
    home_score_col = find_col(df, ["home_score", "home_points", "home_pts"])
    away_score_col = find_col(df, ["away_score", "away_points", "away_pts"])

    if not all([home_team_col, away_team_col, date_col, home_score_col, away_score_col]):
        raise ValueError(
            "Wide-format CBD file missing required columns. "
            "Need home_team/away_team/date/home_score/away_score equivalents."
        )

    source_game_id_col = find_col(df, ["game_id", "event_id", "id", "matchup_id"])
    neutral_col = find_col(df, ["is_neutral", "neutral_site", "neutral"])
    venue_col = find_col(df, ["venue"])
    city_col = find_col(df, ["city"])
    state_col = find_col(df, ["state"])
    tournament_col = find_col(df, ["tournament_name", "tournament"])
    round_col = find_col(df, ["round_name", "round"])
    postseason_col = find_col(df, ["is_postseason", "postseason"])

    prefix_map = {
        "fgm": ("home_fgm", "away_fgm"),
        "fga": ("home_fga", "away_fga"),
        "fg3m": ("home_fg3m", "away_fg3m"),
        "fg3a": ("home_fg3a", "away_fg3a"),
        "ftm": ("home_ftm", "away_ftm"),
        "fta": ("home_fta", "away_fta"),
        "orb": ("home_orb", "away_orb"),
        "drb": ("home_drb", "away_drb"),
        "trb": ("home_trb", "away_trb"),
        "ast": ("home_ast", "away_ast"),
        "stl": ("home_stl", "away_stl"),
        "blk": ("home_blk", "away_blk"),
        "tov": ("home_tov", "away_tov"),
        "pf": ("home_pf", "away_pf"),
    }

    base = pd.DataFrame(
        {
            "source_game_id": df[source_game_id_col].astype(str) if source_game_id_col else None,
            "game_date": pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None),
            "neutral_site": df[neutral_col].map(parse_bool) if neutral_col else False,
            "venue": df[venue_col] if venue_col else None,
            "city": df[city_col] if city_col else None,
            "state": df[state_col] if state_col else None,
            "tournament_name": df[tournament_col] if tournament_col else None,
            "round_name": df[round_col] if round_col else None,
            "is_postseason": df[postseason_col].map(parse_bool) if postseason_col else None,
        }
    )

    home_rows = base.copy()
    home_rows["team_name"] = df[home_team_col].astype(str).str.strip()
    home_rows["opponent_name"] = df[away_team_col].astype(str).str.strip()
    home_rows["points"] = coerce_numeric(df[home_score_col])
    home_rows["opp_points"] = coerce_numeric(df[away_score_col])
    home_rows["is_home"] = True
    home_rows["is_away"] = False

    away_rows = base.copy()
    away_rows["team_name"] = df[away_team_col].astype(str).str.strip()
    away_rows["opponent_name"] = df[home_team_col].astype(str).str.strip()
    away_rows["points"] = coerce_numeric(df[away_score_col])
    away_rows["opp_points"] = coerce_numeric(df[home_score_col])
    away_rows["is_home"] = False
    away_rows["is_away"] = True

    for stat, (home_col, away_col) in prefix_map.items():
        home_rows[stat] = coerce_numeric(df[home_col]) if home_col in df.columns else None
        away_rows[stat] = coerce_numeric(df[away_col]) if away_col in df.columns else None

    out = pd.concat([home_rows, away_rows], ignore_index=True)
    return out


def standardize_cbd_games(df: pd.DataFrame) -> pd.DataFrame:
    long_team_col = find_col(df, ["team_name", "team", "school"])
    long_opp_col = find_col(df, ["opponent_name", "opponent", "opp_name", "opp"])
    if long_team_col and long_opp_col:
        out = standardize_long_format(df)
    else:
        out = standardize_wide_format(df)

    out = out.dropna(subset=["game_date", "team_name", "opponent_name", "points", "opp_points"]).copy()
    out["season"] = SEASON
    out["is_postseason"] = out.apply(infer_postseason, axis=1)
    return out


def assign_game_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_key"] = out.apply(
        lambda r: build_game_key(
            season=SEASON,
            game_date=str(pd.to_datetime(r["game_date"]).date()),
            team1=str(r["team_name"]),
            team2=str(r["opponent_name"]),
            source_game_id=None if pd.isna(r["source_game_id"]) else str(r["source_game_id"]),
        ),
        axis=1,
    )
    return out


def resolve_team_keys(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    alias_map, n_new_teams, n_new_aliases = ensure_team_records(con, df)

    out = df.copy()
    out["team_key"] = out["team_name"].map(lambda x: alias_map.get(normalize_team_name(x)))
    out["opponent_team_key"] = out["opponent_name"].map(lambda x: alias_map.get(normalize_team_name(x)))

    missing = out[out["team_key"].isna() | out["opponent_team_key"].isna()]
    if not missing.empty:
        unresolved = missing[["team_name", "opponent_name"]].drop_duplicates()
        raise RuntimeError(f"Unresolved team names after crosswalk step:\n{unresolved}")

    return out, n_new_teams, n_new_aliases


def keep_complete_games(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("game_key")["team_key"].nunique().reset_index(name="n_teams")
    valid = counts[counts["n_teams"] == 2]["game_key"]
    out = df[df["game_key"].isin(valid)].copy()
    return out


def compute_team_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf"]:
        if col not in out.columns:
            out[col] = None
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["poss_est"] = (
        out["fga"]
        - out["orb"].fillna(0)
        + out["tov"].fillna(0)
        + 0.475 * out["fta"].fillna(0)
    )

    # If possessions are missing because stats are missing, keep null instead of fake 0
    has_poss_inputs = out[["fga", "orb", "tov", "fta"]].notna().any(axis=1)
    out.loc[~has_poss_inputs, "poss_est"] = None

    out["ortg"] = 100.0 * out["points"] / out["poss_est"]
    out["drtg"] = 100.0 * out["opp_points"] / out["poss_est"]
    out["efg"] = (out["fgm"] + 0.5 * out["fg3m"]) / out["fga"]
    out["tov_rate"] = out["tov"] / out["poss_est"]
    out["ftr"] = out["fta"] / out["fga"]
    return out


def add_opponent_metrics(df: pd.DataFrame) -> pd.DataFrame:
    opp = df[
        [
            "game_key",
            "team_key",
            "efg",
            "tov_rate",
            "orb_rate_proxy",
            "ftr",
            "orb",
            "drb",
        ]
    ].copy()
    opp = opp.rename(
        columns={
            "team_key": "opponent_team_key",
            "efg": "opp_efg",
            "tov_rate": "opp_tov_rate",
            "orb_rate_proxy": "opp_orb_rate",
            "ftr": "opp_ftr",
        }
    )

    out = df.merge(opp[["game_key", "opponent_team_key", "opp_efg", "opp_tov_rate", "opp_orb_rate", "opp_ftr"]],
                   on=["game_key", "opponent_team_key"], how="left")
    return out


def finalize_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Offensive rebounding rate proxy from available team + opponent rebound counts
    # ORB% = ORB / (ORB + Opp DRB)
    paired = out[["game_key", "team_key", "opponent_team_key", "orb"]].merge(
        out[["game_key", "team_key", "drb"]].rename(
            columns={"team_key": "opponent_team_key", "drb": "opp_drb"}
        ),
        on=["game_key", "opponent_team_key"],
        how="left",
    )
    paired["orb_rate_proxy"] = paired["orb"] / (paired["orb"] + paired["opp_drb"])
    out = out.merge(
        paired[["game_key", "team_key", "orb_rate_proxy"]],
        on=["game_key", "team_key"],
        how="left",
    )

    out = add_opponent_metrics(out)

    out["result"] = out.apply(
        lambda r: "W" if float(r["points"]) > float(r["opp_points"]) else "L", axis=1
    )

    out["is_neutral"] = out["neutral_site"].fillna(False).astype(bool)

    return out


def build_game_dim_rows(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.sort_values(["game_key", "team_key"]).groupby("game_key", as_index=False)

    rows: list[dict[str, Any]] = []
    for _, g in grouped:
        if len(g) != 2:
            continue

        r1 = g.iloc[0]
        r2 = g.iloc[1]

        winner = r1["team_key"] if r1["points"] > r2["points"] else r2["team_key"]
        loser = r2["team_key"] if winner == r1["team_key"] else r1["team_key"]

        rows.append(
            {
                "game_key": r1["game_key"],
                "season": SEASON,
                "source_game_id": None if pd.isna(r1["source_game_id"]) else str(r1["source_game_id"]),
                "game_date": r1["game_date"],
                "neutral_site": bool(r1["neutral_site"]),
                "venue": r1["venue"],
                "city": r1["city"],
                "state": r1["state"],
                "is_postseason": bool(r1["is_postseason"]),
                "tournament_name": r1["tournament_name"],
                "round_name": r1["round_name"],
                "round_num": None,
                "region": None,
                "slot": None,
                "team1_key": r1["team_key"],
                "team2_key": r2["team_key"],
                "team1_score": int(r1["points"]),
                "team2_score": int(r2["points"]),
                "winner_team_key": winner,
                "loser_team_key": loser,
            }
        )

    return pd.DataFrame(rows)


def upsert_game_dim(con: duckdb.DuckDBPyConnection, game_dim_df: pd.DataFrame) -> int:
    if game_dim_df.empty:
        return 0

    payload = list(
        game_dim_df[
            [
                "game_key",
                "season",
                "source_game_id",
                "game_date",
                "neutral_site",
                "venue",
                "city",
                "state",
                "is_postseason",
                "tournament_name",
                "round_name",
                "round_num",
                "region",
                "slot",
                "team1_key",
                "team2_key",
                "team1_score",
                "team2_score",
                "winner_team_key",
                "loser_team_key",
            ]
        ].itertuples(index=False, name=None)
    )

    con.executemany(
        """
        insert or replace into staging.game_dim
        (game_key, season, source_game_id, game_date, neutral_site, venue, city, state,
         is_postseason, tournament_name, round_name, round_num, region, slot,
         team1_key, team2_key, team1_score, team2_score, winner_team_key, loser_team_key)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def upsert_team_game_stats(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    cols = [
        "game_key",
        "team_key",
        "opponent_team_key",
        "is_home",
        "is_away",
        "is_neutral",
        "result",
        "points",
        "opp_points",
        "fgm",
        "fga",
        "fg3m",
        "fg3a",
        "ftm",
        "fta",
        "orb",
        "drb",
        "trb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
        "poss_est",
        "ortg",
        "drtg",
        "efg",
        "tov_rate",
        "orb_rate_proxy",
        "ftr",
        "opp_efg",
        "opp_tov_rate",
        "opp_orb_rate",
        "opp_ftr",
    ]

    payload = list(df[cols].itertuples(index=False, name=None))

    con.executemany(
        """
        insert or replace into staging.team_game_stats_fact
        (game_key, team_key, opponent_team_key, is_home, is_away, is_neutral,
         result, points, opp_points, fgm, fga, fg3m, fg3a, ftm, fta, orb, drb, trb,
         ast, stl, blk, tov, pf, poss_est, ortg, drtg, efg, tov_rate, orb_rate, ftr,
         opp_efg, opp_tov_rate, opp_orb_rate, opp_ftr)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def main() -> int:
    con = connect_db()
    source_path = locate_or_fetch_source()
    record_snapshot(con, source_path, note="CollegeBasketballData exporter/API import source file.")

    raw_df = load_source_df(source_path)
    standardized = standardize_cbd_games(raw_df)
    standardized = assign_game_keys(standardized)
    standardized, n_new_teams, n_new_aliases = resolve_team_keys(con, standardized)
    standardized = keep_complete_games(standardized)
    standardized = compute_team_metrics(standardized)
    standardized = finalize_advanced_metrics(standardized)

    game_dim_df = build_game_dim_rows(standardized)
    n_games = upsert_game_dim(con, game_dim_df)
    n_team_rows = upsert_team_game_stats(con, standardized)

    print(f"Source file: {source_path}")
    print(f"Raw rows read: {len(raw_df)}")
    print(f"Standardized team-game rows kept: {len(standardized)}")
    print(f"Inserted/updated game_dim rows: {n_games}")
    print(f"Inserted/updated team_game_stats_fact rows: {n_team_rows}")
    print(f"New team_dim rows: {n_new_teams}")
    print(f"New team_name_crosswalk rows: {n_new_aliases}")

    if standardized.empty:
        print("\nNo usable rows were loaded. Check the column names in your CBD export.")
        return 1

    preview = (
        standardized[
            ["game_date", "team_name", "opponent_name", "points", "opp_points", "result"]
        ]
        .sort_values("game_date", ascending=False)
        .head(10)
    )
    print("\nRecent loaded rows:")
    print(preview.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())