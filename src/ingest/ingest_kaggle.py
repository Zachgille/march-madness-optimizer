from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------
# Load env before importing kaggle / kagglehub auth clients
# ---------------------------------------------------------
load_dotenv()


@dataclass
class Settings:
    project_root: Path
    raw_dir: Path
    duckdb_path: Path
    competition: str
    season: int
    source_name: str = "kaggle"


def get_settings() -> Settings:
    project_root = Path(
        os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    ).expanduser()

    raw_dir = project_root / "data" / "raw" / "kaggle"
    duckdb_path = Path(
        os.getenv("DUCKDB_PATH", str(project_root / "data" / "march_madness.duckdb"))
    ).expanduser()

    competition = os.getenv("KAGGLE_COMPETITION", "march-machine-learning-mania-2026")
    season = int(os.getenv("SEASON", "2026"))

    raw_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=project_root,
        raw_dir=raw_dir,
        duckdb_path=duckdb_path,
        competition=competition,
        season=season,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def connect_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.execute("create schema if not exists raw;")
    con.execute("create schema if not exists staging;")
    con.execute("create schema if not exists marts;")
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
    return con


def record_snapshot(
    con: duckdb.DuckDBPyConnection,
    snapshot_id: str,
    source_name: str,
    season: int,
    file_path: Path,
    checksum: str,
    notes: str | None = None,
) -> None:
    con.execute(
        """
        insert or replace into raw.raw_source_snapshot
        (snapshot_id, source_name, fetched_at, season, file_path, checksum, notes)
        values (?, ?, current_timestamp, ?, ?, ?, ?)
        """,
        [snapshot_id, source_name, season, str(file_path), checksum, notes],
    )


def _try_download_with_kagglehub(competition: str, out_dir: Path) -> list[Path]:
    """
    Preferred path when using KAGGLE_API_TOKEN.
    """
    import kagglehub  # installed separately: pip install kagglehub

    # kagglehub uses KAGGLE_API_TOKEN directly when present.
    # The exact function name may evolve, so keep the call isolated here.
    downloaded_paths: list[Path] = []

    # Competition download support exists in KaggleHub; the library places files
    # into a local cache or chosen output directory depending on the API path used.
    # We use the generic competition download handle form here.
    try:
        result = kagglehub.competition_download(competition, path=None)
    except TypeError:
        # Fallback for signature changes
        result = kagglehub.competition_download(competition)

    if isinstance(result, str):
        src = Path(result)
        if src.is_file():
            target = out_dir / src.name
            shutil.copy2(src, target)
            downloaded_paths.append(target)
        elif src.is_dir():
            for p in src.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(src)
                    target = out_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)
                    downloaded_paths.append(target)
    elif isinstance(result, Iterable):
        for item in result:
            p = Path(item)
            if p.exists() and p.is_file():
                target = out_dir / p.name
                shutil.copy2(p, target)
                downloaded_paths.append(target)

    return downloaded_paths


def _write_legacy_credentials_from_token() -> bool:
    """
    Optional compatibility shim:
    If KAGGLE_API_TOKEN contains JSON like {"username":"...","key":"..."},
    export KAGGLE_USERNAME / KAGGLE_KEY so the legacy kaggle package can use them.

    Returns True if legacy env vars were set.
    """
    token = os.getenv("KAGGLE_API_TOKEN", "").strip()
    if not token:
        return False

    # Attempt JSON payload first
    try:
        payload = json.loads(token)
        username = payload.get("username")
        key = payload.get("key")
        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
            return True
    except json.JSONDecodeError:
        pass

    # Attempt "username:key"
    if ":" in token:
        username, key = token.split(":", 1)
        username = username.strip()
        key = key.strip()
        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
            return True

    return False


def _download_with_legacy_kaggle_cli(competition: str, out_dir: Path) -> list[Path]:
    """
    Fallback path for the older kaggle package.
    Supports KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    api.competition_download_files(
        competition=competition,
        path=str(out_dir),
        quiet=False,
        force=False,
    )

    zip_path = out_dir / f"{competition}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected zip not found: {zip_path}")

    return [zip_path]


def unzip_archives(paths: list[Path], out_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    for path in paths:
        if path.suffix.lower() != ".zip":
            continue
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(out_dir)
            extracted.extend(out_dir / name for name in zf.namelist())
    return [p for p in extracted if p.exists()]


def list_csvs(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"])


def canonical_team_key(season: int, team_name: str) -> str:
    import re

    s = team_name.lower().strip()
    s = s.replace("&", "and")
    s = s.replace("st.", "state")
    s = s.replace("saint", "st")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return f"{season}_{s}"


def load_csv_if_exists(root: Path, candidates: list[str]) -> pd.DataFrame | None:
    for name in candidates:
        path = root / name
        if path.exists():
            return pd.read_csv(path)
    return None


def build_team_dim_from_kaggle(extract_dir: Path, season: int) -> pd.DataFrame:
    """
    Handles typical Kaggle MMM naming, but stays flexible across yearly packages.
    """
    teams = load_csv_if_exists(
        extract_dir,
        [
            "MTeams.csv",
            "WTeams.csv",
            "Teams.csv",
        ],
    )

    if teams is None:
        raise FileNotFoundError("Could not find a Teams CSV in extracted Kaggle files.")

    id_col = next((c for c in teams.columns if c.lower() in {"teamid", "team_id"}), None)
    name_col = next((c for c in teams.columns if c.lower() in {"teamname", "team_name"}), None)

    if id_col is None or name_col is None:
        raise ValueError(
            f"Unexpected Teams columns: {teams.columns.tolist()}"
        )

    out = pd.DataFrame(
        {
            "team_key": teams[name_col].map(lambda x: canonical_team_key(season, str(x))),
            "season": season,
            "team_name": teams[name_col].astype(str),
            "short_name": teams[name_col].astype(str),
            "conference": None,
            "division": "D1",
            "kaggle_team_id": teams[id_col].astype("Int64"),
            "cbd_team_id": None,
            "ncaa_team_id": None,
            "yahoo_team_name": None,
            "espn_team_name": None,
            "is_active": True,
        }
    ).drop_duplicates(subset=["team_key"])

    return out


def build_team_name_crosswalk_from_kaggle(extract_dir: Path, season: int) -> pd.DataFrame:
    spellings = load_csv_if_exists(
        extract_dir,
        [
            "MTeamSpellings.csv",
            "WTeamSpellings.csv",
            "TeamSpellings.csv",
        ],
    )

    if spellings is None:
        return pd.DataFrame(
            columns=[
                "alias_name",
                "normalized_alias",
                "source_name",
                "season",
                "team_key",
            ]
        )

    import re

    def norm(s: str) -> str:
        s = s.lower().strip()
        s = s.replace("&", "and")
        s = s.replace("st.", "state")
        s = s.replace("saint", "st")
        s = re.sub(r"[^a-z0-9]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def normalized_colmap(df: pd.DataFrame) -> dict[str, str]:
        out = {}
        for c in df.columns:
            key = re.sub(r"[^a-z0-9]+", "", str(c).lower())
            out[key] = c
        return out

    spellings_colmap = normalized_colmap(spellings)

    id_col = (
        spellings_colmap.get("teamid")
        or spellings_colmap.get("team_id")
    )

    alias_col = (
        spellings_colmap.get("teamnamespelling")
        or spellings_colmap.get("team_name_spelling")
        or spellings_colmap.get("teamnamespelling")
        or spellings_colmap.get("spelling")
    )

    if id_col is None or alias_col is None:
        raise ValueError(
            f"Unexpected TeamSpellings columns: {spellings.columns.tolist()}"
        )

    teams = load_csv_if_exists(extract_dir, ["MTeams.csv", "WTeams.csv", "Teams.csv"])
    if teams is None:
        raise FileNotFoundError("Could not find Teams CSV while building team spellings crosswalk.")

    teams_colmap = normalized_colmap(teams)

    team_id_col = (
        teams_colmap.get("teamid")
        or teams_colmap.get("team_id")
    )
    team_name_col = (
        teams_colmap.get("teamname")
        or teams_colmap.get("team_name")
    )

    if team_id_col is None or team_name_col is None:
        raise ValueError(
            f"Unexpected Teams columns: {teams.columns.tolist()}"
        )

    merged = spellings.merge(
        teams[[team_id_col, team_name_col]],
        left_on=id_col,
        right_on=team_id_col,
        how="left",
    )

    out = pd.DataFrame(
        {
            "alias_name": merged[alias_col].astype(str),
            "normalized_alias": merged[alias_col].astype(str).map(norm),
            "source_name": "kaggle",
            "season": season,
            "team_key": merged[team_name_col].astype(str).map(lambda x: canonical_team_key(season, x)),
        }
    ).drop_duplicates()

    return out


def upsert_dataframe(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_name: str,
) -> None:
    temp_name = f"tmp_{table_name.split('.')[-1]}"
    con.register(temp_name, df)
    con.execute(f"create or replace table {table_name} as select * from {temp_name}")
    con.unregister(temp_name)


def main() -> int:
    settings = get_settings()
    con = connect_db(settings.duckdb_path)

    snapshot_suffix = utc_now_iso()
    extract_dir = settings.raw_dir / f"extract_{snapshot_suffix}"
    extract_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[Path] = []

    token_present = bool(os.getenv("KAGGLE_API_TOKEN", "").strip())
    legacy_user_present = bool(os.getenv("KAGGLE_USERNAME", "").strip())
    legacy_key_present = bool(os.getenv("KAGGLE_KEY", "").strip())

    # Preferred: kagglehub with KAGGLE_API_TOKEN
    if token_present:
        try:
            print("Using kagglehub with KAGGLE_API_TOKEN...")
            downloaded_files = _try_download_with_kagglehub(settings.competition, extract_dir)
        except Exception as e:
            print(f"kagglehub path failed: {e}", file=sys.stderr)

    # Fallback: old kaggle client, optionally fed by KAGGLE_API_TOKEN if it contains legacy JSON
    if not downloaded_files:
        if not (legacy_user_present and legacy_key_present):
            _write_legacy_credentials_from_token()

        try:
            print("Using legacy kaggle client fallback...")
            downloaded_files = _download_with_legacy_kaggle_cli(settings.competition, extract_dir)
        except Exception as e:
            print(f"legacy kaggle path failed: {e}", file=sys.stderr)
            return 1

    if not downloaded_files:
        print("No files were downloaded from Kaggle.", file=sys.stderr)
        return 1

    # If we downloaded zip archives, extract them.
    extracted_files = unzip_archives(downloaded_files, extract_dir)

    # Record raw snapshots
    for path in downloaded_files:
        if path.exists():
            snapshot_id = f"{settings.source_name}_{snapshot_suffix}_{path.name}"
            record_snapshot(
                con=con,
                snapshot_id=snapshot_id,
                source_name=settings.source_name,
                season=settings.season,
                file_path=path,
                checksum=sha256_file(path),
                notes="Downloaded from Kaggle competition source.",
            )

    for path in extracted_files:
        if path.exists() and path.is_file():
            snapshot_id = f"{settings.source_name}_{snapshot_suffix}_{path.name}"
            record_snapshot(
                con=con,
                snapshot_id=snapshot_id,
                source_name=settings.source_name,
                season=settings.season,
                file_path=path,
                checksum=sha256_file(path),
                notes="Extracted from Kaggle archive.",
            )

    csvs = list_csvs(extract_dir)
    if not csvs:
        print(f"No CSV files found in {extract_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(csvs)} CSV files in {extract_dir}")

    # Build starter staging tables
    team_dim = build_team_dim_from_kaggle(extract_dir, settings.season)
    team_name_crosswalk = build_team_name_crosswalk_from_kaggle(extract_dir, settings.season)

    upsert_dataframe(con, team_dim, "staging.team_dim")
    upsert_dataframe(con, team_name_crosswalk, "staging.team_name_crosswalk")

    manifest = pd.DataFrame(
        {
            "file_name": [p.name for p in csvs],
            "file_path": [str(p) for p in csvs],
            "n_bytes": [p.stat().st_size for p in csvs],
        }
    )
    upsert_dataframe(con, manifest, "raw.kaggle_file_manifest")

    print("Ingest complete.")
    print(f"DuckDB: {settings.duckdb_path}")
    print(f"Extracted files: {extract_dir}")
    print(f"team_dim rows: {len(team_dim)}")
    print(f"team_name_crosswalk rows: {len(team_name_crosswalk)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())