from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

from src.utils.ids import normalize_team_name

load_dotenv()

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser()
DUCKDB_PATH = Path(
    os.getenv("DUCKDB_PATH", str(PROJECT_ROOT / "data" / "march_madness.duckdb"))
).expanduser()
SEASON = int(os.getenv("SEASON", "2026"))

PUBLIC_PICKS_PATH = PROJECT_ROOT / "manual" / "public_picks_2026.csv"


REQUIRED_COLUMNS = {
    "season",
    "source_name",
    "round_num",
    "team_name",
    "pct_pick_to_advance",
}


def connect_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("create schema if not exists staging;")
    con.execute(
        """
        create table if not exists staging.public_pick_fact (
            season integer not null,
            source_name varchar not null,
            snapshot_ts timestamp not null,
            round_num integer not null,
            team_key varchar not null,
            pct_pick_to_advance double,
            pct_pick_to_reach_round double,
            pct_pick_to_win_title double,
            primary key (season, source_name, snapshot_ts, round_num, team_key)
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
    return con


def parse_pct(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    s = str(value).strip()
    if not s:
        return None

    if s.endswith("%"):
        return float(s[:-1].strip()) / 100.0

    x = float(s)
    if x > 1:
        return x / 100.0
    return x


def load_public_picks_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["season"] = df["season"].astype(int)
    df = df[df["season"] == SEASON].copy()

    if df.empty:
        raise ValueError(f"No rows found for season {SEASON} in {path.name}")

    df["source_name"] = df["source_name"].astype(str).str.strip().str.lower()
    df["round_num"] = df["round_num"].astype(int)
    df["team_name"] = df["team_name"].astype(str).str.strip()
    df["normalized_team_name"] = df["team_name"].map(normalize_team_name)

    for col in [
        "pct_pick_to_advance",
        "pct_pick_to_reach_round",
        "pct_pick_to_win_title",
    ]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].map(parse_pct)

    if "snapshot_ts" not in df.columns:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        df["snapshot_ts"] = now_utc

    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=False)

    return df


def load_alias_map(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    df = con.execute(
        """
        select normalized_alias, team_key
        from staging.team_name_crosswalk
        where season = ?
        order by
            case source_name
                when 'manual' then 1
                when 'public_picks' then 2
                when 'ncaa_bracket' then 3
                when 'canonical' then 4
                else 5
            end
        """,
        [SEASON],
    ).df()

    alias_map: dict[str, str] = {}
    for _, row in df.iterrows():
        alias_map.setdefault(row["normalized_alias"], row["team_key"])
    return alias_map


def resolve_team_keys(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    alias_map = load_alias_map(con)

    team_keys: list[str] = []
    unmapped: list[str] = []

    for _, row in df.iterrows():
        key = alias_map.get(row["normalized_team_name"])
        team_keys.append(key)
        if key is None:
            unmapped.append(row["team_name"])

    out = df.copy()
    out["team_key"] = team_keys
    out = out[out["team_key"].notna()].copy()

    return out, sorted(set(unmapped))


def upsert_public_picks(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    payload = list(
        df[
            [
                "season",
                "source_name",
                "snapshot_ts",
                "round_num",
                "team_key",
                "pct_pick_to_advance",
                "pct_pick_to_reach_round",
                "pct_pick_to_win_title",
            ]
        ].itertuples(index=False, name=None)
    )

    con.executemany(
        """
        insert or replace into staging.public_pick_fact
        (season, source_name, snapshot_ts, round_num, team_key,
         pct_pick_to_advance, pct_pick_to_reach_round, pct_pick_to_win_title)
        values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def main() -> int:
    con = connect_db()
    df = load_public_picks_csv(PUBLIC_PICKS_PATH)
    resolved_df, unmapped = resolve_team_keys(con, df)

    if unmapped:
        print("Unmapped public-pick team names:")
        for name in unmapped:
            print(f"  - {name}")
        print("\nAdd them to manual/team_crosswalk_overrides.csv and rerun.")

    n = upsert_public_picks(con, resolved_df)

    print(f"Inserted/updated public pick rows: {n}")
    print(f"Resolved rows: {len(resolved_df)} / {len(df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())