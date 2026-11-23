from __future__ import annotations

import os
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

MANUAL_CROSSWALK_PATH = PROJECT_ROOT / "manual" / "team_crosswalk_overrides.csv"
BRACKET_PATH = PROJECT_ROOT / "manual" / "2026_mens_bracket.csv"
PUBLIC_PICKS_PATH = PROJECT_ROOT / "manual" / "public_picks_2026.csv"


def connect_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("create schema if not exists staging;")
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


def require_team_dim(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    tables = con.execute(
        """
        select count(*) as n
        from information_schema.tables
        where table_schema = 'staging' and table_name = 'team_dim'
        """
    ).df()

    if tables.loc[0, "n"] == 0:
        raise RuntimeError(
            "staging.team_dim does not exist yet. Run schema.sql and ingest_kaggle.py first."
        )

    df = con.execute("select * from staging.team_dim").df()
    if df.empty:
        raise RuntimeError("staging.team_dim is empty. Run ingest_kaggle.py first.")
    return df


def build_canonical_rows(team_dim: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for _, row in team_dim.iterrows():
        aliases = {str(row["team_name"]).strip()}
        short_name = row.get("short_name")
        if pd.notna(short_name) and str(short_name).strip():
            aliases.add(str(short_name).strip())

        for alias in aliases:
            rows.append(
                {
                    "alias_name": alias,
                    "normalized_alias": normalize_team_name(alias),
                    "source_name": "canonical",
                    "season": int(row["season"]),
                    "team_key": row["team_key"],
                }
            )

    return pd.DataFrame(rows).drop_duplicates()


def load_manual_overrides() -> pd.DataFrame:
    if not MANUAL_CROSSWALK_PATH.exists():
        return pd.DataFrame(
            columns=["source_name", "alias_name", "team_key", "season"]
        )

    df = pd.read_csv(MANUAL_CROSSWALK_PATH)
    required = {"source_name", "alias_name", "team_key", "season"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{MANUAL_CROSSWALK_PATH} is missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["season"] = df["season"].fillna(SEASON).astype(int)
    df["alias_name"] = df["alias_name"].astype(str).str.strip()
    df["normalized_alias"] = df["alias_name"].map(normalize_team_name)

    return df[
        ["alias_name", "normalized_alias", "source_name", "season", "team_key"]
    ].drop_duplicates()


def upsert_crosswalk_rows(
    con: duckdb.DuckDBPyConnection,
    rows: pd.DataFrame,
) -> int:
    if rows.empty:
        return 0

    payload = list(
        rows[
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


def current_alias_map(con: duckdb.DuckDBPyConnection, season: int) -> dict[str, str]:
    df = con.execute(
        """
        select normalized_alias, team_key
        from staging.team_name_crosswalk
        where season = ?
        """,
        [season],
    ).df()

    # keep first occurrence if duplicates exist
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        out.setdefault(row["normalized_alias"], row["team_key"])
    return out


def print_unmapped_from_file(
    alias_map: dict[str, str],
    path: Path,
    team_col: str,
    label: str,
) -> None:
    if not path.exists():
        return

    df = pd.read_csv(path)
    if team_col not in df.columns:
        print(f"[warn] {label}: column '{team_col}' not found in {path.name}")
        return

    names = sorted({str(x).strip() for x in df[team_col].dropna().tolist()})
    unmapped = [name for name in names if normalize_team_name(name) not in alias_map]

    if unmapped:
        print(f"\nUnmapped aliases from {label}:")
        for name in unmapped:
            print(f"  - {name}")
        print("\nAdd these to manual/team_crosswalk_overrides.csv if needed.")
    else:
        print(f"\nAll {label} team names are currently mapped.")


def main() -> int:
    con = connect_db()
    team_dim = require_team_dim(con)

    canonical_rows = build_canonical_rows(team_dim)
    manual_rows = load_manual_overrides()

    n1 = upsert_crosswalk_rows(con, canonical_rows)
    n2 = upsert_crosswalk_rows(con, manual_rows)

    alias_map = current_alias_map(con, SEASON)

    print(f"Upserted {n1} canonical crosswalk rows.")
    print(f"Upserted {n2} manual override rows.")

    print_unmapped_from_file(
        alias_map=alias_map,
        path=BRACKET_PATH,
        team_col="team_name",
        label="manual bracket file",
    )

    print_unmapped_from_file(
        alias_map=alias_map,
        path=PUBLIC_PICKS_PATH,
        team_col="team_name",
        label="public picks file",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())