from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

from src.utils.ids import (
    canonical_team_key,
    normalize_team_name,
    parse_bool,
    parse_seed_label,
)

load_dotenv()


PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser()
DUCKDB_PATH = Path(
    os.getenv("DUCKDB_PATH", str(PROJECT_ROOT / "data" / "march_madness.duckdb"))
).expanduser()
SEASON = int(os.getenv("SEASON", "2026"))

BRACKET_PATH = PROJECT_ROOT / "manual" / "2026_mens_bracket.csv"


REQUIRED_COLUMNS = {
    "season",
    "region",
    "seed_label",
    "team_name",
    "is_play_in",
    "play_in_group",
    "initial_slot",
}


def connect_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("create schema if not exists staging;")
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
        create table if not exists staging.tournament_seed_dim (
            season integer not null,
            team_key varchar not null,
            region varchar not null,
            seed integer not null,
            seed_label varchar not null,
            play_in_group varchar,
            is_play_in boolean default false,
            initial_slot varchar,
            primary key (season, team_key)
        );
        """
    )
    return con


def load_bracket_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Create the file first."
        )

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["season"] = df["season"].astype(int)
    df = df[df["season"] == SEASON].copy()

    if df.empty:
        raise ValueError(f"No rows found for season {SEASON} in {path.name}")

    for col in ["region", "seed_label", "team_name", "initial_slot"]:
        df[col] = df[col].astype(str).str.strip()

    df["play_in_group"] = df["play_in_group"].where(
        df["play_in_group"].notna(), None
    )
    df["is_play_in"] = df["is_play_in"].map(parse_bool)

    parsed = df["seed_label"].map(parse_seed_label)
    df["seed"] = parsed.map(lambda x: x[0])

    # If seed_label implies play-in and CSV is missing it, fill it
    implied_is_play_in = parsed.map(lambda x: x[1])
    implied_group = parsed.map(lambda x: x[2])

    df["is_play_in"] = df["is_play_in"] | implied_is_play_in
    df["play_in_group"] = df["play_in_group"].fillna(implied_group)

    df["normalized_team_name"] = df["team_name"].map(normalize_team_name)

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
                when 'ncaa_bracket' then 2
                when 'canonical' then 3
                else 4
            end
        """,
        [SEASON],
    ).df()

    out: dict[str, str] = {}
    for _, row in df.iterrows():
        out.setdefault(row["normalized_alias"], row["team_key"])
    return out


def resolve_team_keys(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alias_map = load_alias_map(con)

    new_teams: list[dict] = []
    bracket_crosswalk_rows: list[dict] = []
    resolved_keys: list[str] = []

    existing_team_dim = con.execute(
        "select team_key from staging.team_dim where season = ?",
        [SEASON],
    ).df()
    existing_keys = set(existing_team_dim["team_key"].tolist())

    for _, row in df.iterrows():
        team_name = row["team_name"]
        normalized = row["normalized_team_name"]

        team_key = alias_map.get(normalized)
        if not team_key:
            team_key = canonical_team_key(SEASON, team_name)

        if team_key not in existing_keys:
            new_teams.append(
                {
                    "team_key": team_key,
                    "season": SEASON,
                    "team_name": team_name,
                    "short_name": team_name,
                    "conference": None,
                    "division": "D1",
                    "kaggle_team_id": None,
                    "cbd_team_id": None,
                    "ncaa_team_id": None,
                    "yahoo_team_name": None,
                    "espn_team_name": None,
                    "is_active": True,
                }
            )
            existing_keys.add(team_key)

        bracket_crosswalk_rows.append(
            {
                "alias_name": team_name,
                "normalized_alias": normalized,
                "source_name": "ncaa_bracket",
                "season": SEASON,
                "team_key": team_key,
            }
        )
        resolved_keys.append(team_key)

    out = df.copy()
    out["team_key"] = resolved_keys

    new_teams_df = pd.DataFrame(new_teams).drop_duplicates(subset=["team_key"])
    bracket_crosswalk_df = pd.DataFrame(bracket_crosswalk_rows).drop_duplicates()

    return out, new_teams_df, bracket_crosswalk_df


def upsert_team_dim(con: duckdb.DuckDBPyConnection, teams_df: pd.DataFrame) -> int:
    if teams_df.empty:
        return 0

    payload = list(
        teams_df[
            [
                "team_key",
                "season",
                "team_name",
                "short_name",
                "conference",
                "division",
                "kaggle_team_id",
                "cbd_team_id",
                "ncaa_team_id",
                "yahoo_team_name",
                "espn_team_name",
                "is_active",
            ]
        ].itertuples(index=False, name=None)
    )

    con.executemany(
        """
        insert or replace into staging.team_dim
        (team_key, season, team_name, short_name, conference, division,
         kaggle_team_id, cbd_team_id, ncaa_team_id, yahoo_team_name, espn_team_name, is_active)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def upsert_crosswalk(con: duckdb.DuckDBPyConnection, rows_df: pd.DataFrame) -> int:
    if rows_df.empty:
        return 0

    payload = list(
        rows_df[
            ["alias_name", "normalized_alias", "source_name", "season", "team_key"]
        ].itertuples(index=False, name=None)
    )
    con.executemany(
        """
        insert or replace into staging.team_name_crosswalk
        (alias_name, normalized_alias, source_name, season, team_key)
        values (?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def replace_tournament_seeds(
    con: duckdb.DuckDBPyConnection,
    seeds_df: pd.DataFrame,
) -> int:
    con.execute("delete from staging.tournament_seed_dim where season = ?", [SEASON])

    payload = list(
        seeds_df[
            [
                "season",
                "team_key",
                "region",
                "seed",
                "seed_label",
                "play_in_group",
                "is_play_in",
                "initial_slot",
            ]
        ].itertuples(index=False, name=None)
    )

    con.executemany(
        """
        insert into staging.tournament_seed_dim
        (season, team_key, region, seed, seed_label, play_in_group, is_play_in, initial_slot)
        values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def validate_bracket(df: pd.DataFrame) -> None:
    if df["team_key"].duplicated().any():
        dupes = df.loc[df["team_key"].duplicated(), ["team_name", "team_key"]]
        raise ValueError(f"Duplicate team_key values found:\n{dupes}")

    n_rows = len(df)
    if n_rows != 68:
        print(f"[warn] Expected 68 rows, found {n_rows}.")
        print("       This is allowed temporarily, but a full men’s field should have 68 teams.")

    regions = sorted(df["region"].dropna().unique().tolist())
    print(f"Regions found: {regions}")

    seed_summary = (
        df.groupby(["region", "seed"])["team_key"]
        .count()
        .reset_index(name="n_teams")
        .sort_values(["region", "seed"])
    )
    print("\nSeed counts by region:")
    print(seed_summary.to_string(index=False))

    play_in_summary = (
        df[df["is_play_in"]]
        .groupby(["region", "play_in_group"])["team_key"]
        .count()
        .reset_index(name="n_teams")
        .sort_values(["region", "play_in_group"])
    )

    if not play_in_summary.empty:
        print("\nPlay-in groups:")
        print(play_in_summary.to_string(index=False))


def main() -> int:
    con = connect_db()
    bracket_df = load_bracket_csv(BRACKET_PATH)
    bracket_df, new_teams_df, bracket_crosswalk_df = resolve_team_keys(con, bracket_df)

    n_new_teams = upsert_team_dim(con, new_teams_df)
    n_crosswalk = upsert_crosswalk(con, bracket_crosswalk_df)
    n_seeds = replace_tournament_seeds(con, bracket_df)

    validate_bracket(bracket_df)

    print("\nIngest complete.")
    print(f"Inserted/updated new team_dim rows: {n_new_teams}")
    print(f"Inserted/updated ncaa_bracket crosswalk rows: {n_crosswalk}")
    print(f"Inserted tournament_seed_dim rows: {n_seeds}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())