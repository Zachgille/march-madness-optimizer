from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests
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
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()

SPORT_KEY = os.getenv("ODDS_SPORT_KEY", "basketball_ncaab")
REGIONS = os.getenv("ODDS_REGIONS", "us")
MARKETS = os.getenv("ODDS_MARKETS", "h2h,spreads,totals")
ODDS_FORMAT = os.getenv("ODDS_FORMAT", "american")
DATE_FORMAT = os.getenv("ODDS_DATE_FORMAT", "iso")

RAW_OUT_DIR = PROJECT_ROOT / "data" / "raw" / "odds"


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
        create table if not exists staging.market_odds_fact (
            odds_row_id varchar primary key,
            season integer not null,
            game_key varchar,
            source_name varchar not null,
            bookmaker varchar not null,
            market_type varchar not null,
            snapshot_ts timestamp not null,
            team1_key varchar,
            team2_key varchar,
            price_team1 double,
            price_team2 double,
            spread_team1 double,
            spread_team2 double,
            total_points double,
            implied_prob_team1_raw double,
            implied_prob_team2_raw double,
            implied_prob_team1_novig double,
            implied_prob_team2_novig double
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


def moneyline_to_implied_prob(price: float | None) -> float | None:
    if price is None or pd.isna(price):
        return None

    price = float(price)
    if price > 0:
        return 100.0 / (price + 100.0)
    if price < 0:
        return (-price) / ((-price) + 100.0)
    return None


def remove_vig(p1_raw: float | None, p2_raw: float | None) -> tuple[float | None, float | None]:
    if p1_raw is None or p2_raw is None:
        return None, None
    denom = p1_raw + p2_raw
    if denom <= 0:
        return None, None
    return p1_raw / denom, p2_raw / denom


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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

    alias_map: dict[str, str] = {}
    for _, row in df.iterrows():
        alias_map.setdefault(row["normalized_alias"], row["team_key"])
    return alias_map


def fetch_odds() -> tuple[list[dict[str, Any]], str]:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY is not set yet.")

    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json(), resp.text


def save_raw_snapshot(con: duckdb.DuckDBPyConnection, raw_text: str) -> Path:
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = utc_now_str()
    out_path = RAW_OUT_DIR / f"odds_snapshot_{ts}.json"
    out_path.write_text(raw_text, encoding="utf-8")

    con.execute(
        """
        insert or replace into raw.raw_source_snapshot
        (snapshot_id, source_name, fetched_at, season, file_path, checksum, notes)
        values (?, 'odds_api', current_timestamp, ?, ?, ?, ?)
        """,
        [
            f"odds_api_{ts}",
            SEASON,
            str(out_path),
            sha256_text(raw_text),
            f"sport={SPORT_KEY}; markets={MARKETS}; regions={REGIONS}",
        ],
    )
    return out_path


def get_outcome_map(market: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outcomes = market.get("outcomes", []) or []
    out: dict[str, dict[str, Any]] = {}
    for item in outcomes:
        name = item.get("name")
        if name:
            out[str(name)] = item
    return out


def build_market_rows(
    events: list[dict[str, Any]],
    alias_map: dict[str, str],
    snapshot_ts: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for event in events:
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")

        if not home_team or not away_team:
            continue

        team1_name = str(home_team).strip()
        team2_name = str(away_team).strip()

        team1_key = alias_map.get(normalize_team_name(team1_name))
        team2_key = alias_map.get(normalize_team_name(team2_name))

        bookmakers = event.get("bookmakers", []) or []
        for bookmaker in bookmakers:
            bookmaker_name = bookmaker.get("key") or bookmaker.get("title") or "unknown_book"

            for market in bookmaker.get("markets", []) or []:
                market_type = market.get("key", "")
                outcome_map = get_outcome_map(market)

                row = {
                    "season": SEASON,
                    "game_key": None,
                    "source_name": "odds_api",
                    "bookmaker": bookmaker_name,
                    "market_type": market_type,
                    "snapshot_ts": snapshot_ts,
                    "team1_key": team1_key,
                    "team2_key": team2_key,
                    "price_team1": None,
                    "price_team2": None,
                    "spread_team1": None,
                    "spread_team2": None,
                    "total_points": None,
                    "implied_prob_team1_raw": None,
                    "implied_prob_team2_raw": None,
                    "implied_prob_team1_novig": None,
                    "implied_prob_team2_novig": None,
                }

                if market_type == "h2h":
                    o1 = outcome_map.get(team1_name)
                    o2 = outcome_map.get(team2_name)
                    p1 = float(o1["price"]) if o1 and o1.get("price") is not None else None
                    p2 = float(o2["price"]) if o2 and o2.get("price") is not None else None

                    p1_raw = moneyline_to_implied_prob(p1)
                    p2_raw = moneyline_to_implied_prob(p2)
                    p1_nv, p2_nv = remove_vig(p1_raw, p2_raw)

                    row["price_team1"] = p1
                    row["price_team2"] = p2
                    row["implied_prob_team1_raw"] = p1_raw
                    row["implied_prob_team2_raw"] = p2_raw
                    row["implied_prob_team1_novig"] = p1_nv
                    row["implied_prob_team2_novig"] = p2_nv

                elif market_type == "spreads":
                    o1 = outcome_map.get(team1_name)
                    o2 = outcome_map.get(team2_name)
                    row["spread_team1"] = (
                        float(o1["point"]) if o1 and o1.get("point") is not None else None
                    )
                    row["spread_team2"] = (
                        float(o2["point"]) if o2 and o2.get("point") is not None else None
                    )
                    row["price_team1"] = (
                        float(o1["price"]) if o1 and o1.get("price") is not None else None
                    )
                    row["price_team2"] = (
                        float(o2["price"]) if o2 and o2.get("price") is not None else None
                    )

                elif market_type == "totals":
                    over = outcome_map.get("Over")
                    under = outcome_map.get("Under")
                    if over and over.get("point") is not None:
                        row["total_points"] = float(over["point"])
                    row["price_team1"] = (
                        float(over["price"]) if over and over.get("price") is not None else None
                    )
                    row["price_team2"] = (
                        float(under["price"]) if under and under.get("price") is not None else None
                    )

                else:
                    continue

                hash_input = "|".join(
                    [
                        str(SEASON),
                        str(commence_time),
                        str(bookmaker_name),
                        str(market_type),
                        str(snapshot_ts),
                        str(team1_key or team1_name),
                        str(team2_key or team2_name),
                    ]
                )
                row["odds_row_id"] = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
                rows.append(row)

    return pd.DataFrame(rows)


def upsert_market_rows(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    payload = list(
        df[
            [
                "odds_row_id",
                "season",
                "game_key",
                "source_name",
                "bookmaker",
                "market_type",
                "snapshot_ts",
                "team1_key",
                "team2_key",
                "price_team1",
                "price_team2",
                "spread_team1",
                "spread_team2",
                "total_points",
                "implied_prob_team1_raw",
                "implied_prob_team2_raw",
                "implied_prob_team1_novig",
                "implied_prob_team2_novig",
            ]
        ].itertuples(index=False, name=None)
    )

    con.executemany(
        """
        insert or replace into staging.market_odds_fact
        (odds_row_id, season, game_key, source_name, bookmaker, market_type, snapshot_ts,
         team1_key, team2_key, price_team1, price_team2, spread_team1, spread_team2,
         total_points, implied_prob_team1_raw, implied_prob_team2_raw,
         implied_prob_team1_novig, implied_prob_team2_novig)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def main() -> int:
    con = connect_db()
    alias_map = load_alias_map(con)

    try:
        events, raw_text = fetch_odds()
    except RuntimeError as e:
        print(str(e))
        print("Leave this file in place and rerun after your ODDS_API_KEY arrives.")
        return 0

    save_path = save_raw_snapshot(con, raw_text)
    snapshot_ts = pd.Timestamp.utcnow().floor("s")
    rows = build_market_rows(events, alias_map, snapshot_ts=snapshot_ts)

    unresolved_games = rows[
        rows["team1_key"].isna() | rows["team2_key"].isna()
    ][["team1_key", "team2_key"]].shape[0]

    n = upsert_market_rows(con, rows)

    print(f"Saved raw odds snapshot to: {save_path}")
    print(f"Inserted/updated market_odds_fact rows: {n}")
    print(f"Rows with unmapped teams: {unresolved_games}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())