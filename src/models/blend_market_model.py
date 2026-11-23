from __future__ import annotations

import hashlib
import os
from pathlib import Path

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

INPUT_MODEL_VERSION = os.getenv("INPUT_MODEL_VERSION", "logistic_seed_margin_v1").strip()
OUTPUT_MODEL_VERSION = os.getenv(
    "OUTPUT_MODEL_VERSION",
    f"{INPUT_MODEL_VERSION}_blend_v1",
).strip()

BLEND_METHOD = os.getenv("BLEND_METHOD", "logit").strip().lower()
BLEND_MARKET_WEIGHT = float(os.getenv("BLEND_MARKET_WEIGHT", "0.70"))
BLEND_MODEL_WEIGHT = float(os.getenv("BLEND_MODEL_WEIGHT", "0.30"))
BLEND_INTERCEPT = float(os.getenv("BLEND_INTERCEPT", "0.0"))
BLEND_ALPHA = float(os.getenv("BLEND_ALPHA", "0.70"))  # used only for convex blend
EPSILON = float(os.getenv("BLEND_EPSILON", "1e-6"))


def connect_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("create schema if not exists staging;")
    con.execute("create schema if not exists marts;")

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
        create table if not exists marts.matchup_prediction_fact (
            prediction_id varchar primary key,
            model_version varchar not null,
            season integer not null,
            as_of_date date not null,
            team_a_key varchar not null,
            team_b_key varchar not null,
            p_model_a double not null,
            p_market_a double,
            p_blend_a double not null,
            calibration_bucket varchar,
            created_at timestamp not null
        );
        """
    )

    return con


def clip_prob(p: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(p, EPSILON, 1.0 - EPSILON)


def logit(p: pd.Series) -> pd.Series:
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def logistic(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def calibration_bucket(p: float) -> str:
    bucket = int(np.floor(max(0.0, min(0.999999, p)) * 20)) / 20
    return f"{bucket:.2f}-{bucket + 0.05:.2f}"


def get_latest_prediction_date(
    con: duckdb.DuckDBPyConnection,
    season: int,
    model_version: str,
) -> str:
    row = con.execute(
        """
        select cast(max(as_of_date) as varchar)
        from marts.matchup_prediction_fact
        where season = ? and model_version = ?
        """,
        [season, model_version],
    ).fetchone()

    if row is None or row[0] is None:
        raise RuntimeError(
            f"No matchup predictions found for season={season}, model_version={model_version}."
        )
    return str(row[0])


def load_input_predictions(
    con: duckdb.DuckDBPyConnection,
    season: int,
    model_version: str,
    as_of_date: str,
) -> pd.DataFrame:
    df = con.execute(
        """
        select
            prediction_id,
            model_version,
            season,
            as_of_date,
            team_a_key,
            team_b_key,
            p_model_a,
            p_market_a,
            p_blend_a,
            calibration_bucket,
            created_at
        from marts.matchup_prediction_fact
        where season = ?
          and model_version = ?
          and as_of_date = ?
        """,
        [season, model_version, as_of_date],
    ).df()

    if df.empty:
        raise RuntimeError(
            f"No prediction rows found for season={season}, model_version={model_version}, as_of_date={as_of_date}."
        )

    return df


def load_latest_market_rows(
    con: duckdb.DuckDBPyConnection,
    season: int,
) -> pd.DataFrame:
    df = con.execute(
        """
        select
            odds_row_id,
            season,
            bookmaker,
            market_type,
            snapshot_ts,
            team1_key,
            team2_key,
            spread_team1,
            spread_team2,
            total_points,
            implied_prob_team1_novig,
            implied_prob_team2_novig
        from staging.market_odds_fact
        where season = ?
          and market_type = 'h2h'
          and team1_key is not null
          and team2_key is not null
          and implied_prob_team1_novig is not null
          and implied_prob_team2_novig is not null
        """,
        [season],
    ).df()

    if df.empty:
        return df

    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    df["pair_low"] = df[["team1_key", "team2_key"]].min(axis=1)
    df["pair_high"] = df[["team1_key", "team2_key"]].max(axis=1)

    # keep only the latest row per bookmaker and unordered pair
    df = (
        df.sort_values("snapshot_ts")
        .groupby(["bookmaker", "pair_low", "pair_high"], as_index=False)
        .tail(1)
        .copy()
    )

    return df


def build_directional_market_probs(market_df: pd.DataFrame) -> pd.DataFrame:
    if market_df.empty:
        return pd.DataFrame(
            columns=[
                "team_a_key",
                "team_b_key",
                "p_market_a",
                "n_books",
            ]
        )

    d1 = market_df[
        ["team1_key", "team2_key", "implied_prob_team1_novig"]
    ].rename(
        columns={
            "team1_key": "team_a_key",
            "team2_key": "team_b_key",
            "implied_prob_team1_novig": "p_market_a",
        }
    )

    d2 = market_df[
        ["team2_key", "team1_key", "implied_prob_team2_novig"]
    ].rename(
        columns={
            "team2_key": "team_a_key",
            "team1_key": "team_b_key",
            "implied_prob_team2_novig": "p_market_a",
        }
    )

    combined = pd.concat([d1, d2], ignore_index=True)
    combined["p_market_a"] = pd.to_numeric(combined["p_market_a"], errors="coerce")

    out = (
        combined.groupby(["team_a_key", "team_b_key"], as_index=False)
        .agg(
            p_market_a=("p_market_a", "mean"),
            n_books=("p_market_a", "count"),
        )
    )

    return out


def apply_blend(pred_df: pd.DataFrame, market_dir_df: pd.DataFrame) -> pd.DataFrame:
    # Drop previously stored market/blend outputs from the input predictions so the
    # merge brings in a single clean p_market_a column from latest market data.
    out = pred_df.drop(
        columns=["p_market_a", "p_blend_a", "calibration_bucket"],
        errors="ignore",
    ).merge(
        market_dir_df,
        on=["team_a_key", "team_b_key"],
        how="left",
    )

    # Ensure n_books always exists for downstream preview/summarization.
    if "n_books" not in out.columns:
        out["n_books"] = np.nan

    out["p_model_a"] = pd.to_numeric(out["p_model_a"], errors="coerce")
    out["p_market_a"] = pd.to_numeric(out["p_market_a"], errors="coerce")
    out["n_books"] = pd.to_numeric(out["n_books"], errors="coerce")

    if BLEND_METHOD == "convex":
        out["p_blend_a"] = np.where(
            out["p_market_a"].notna(),
            BLEND_ALPHA * clip_prob(out["p_market_a"])
            + (1.0 - BLEND_ALPHA) * clip_prob(out["p_model_a"]),
            clip_prob(out["p_model_a"]),
        )
    else:
        has_market = out["p_market_a"].notna()
        out["p_blend_a"] = clip_prob(out["p_model_a"])

        if has_market.any():
            model_logit = logit(out.loc[has_market, "p_model_a"])
            market_logit = logit(out.loc[has_market, "p_market_a"])
            blend_logit = (
                BLEND_MARKET_WEIGHT * market_logit
                + BLEND_MODEL_WEIGHT * model_logit
                + BLEND_INTERCEPT
            )
            out.loc[has_market, "p_blend_a"] = logistic(blend_logit)

    out["p_blend_a"] = clip_prob(out["p_blend_a"])
    out["calibration_bucket"] = out["p_blend_a"].map(calibration_bucket)

    out["model_version"] = OUTPUT_MODEL_VERSION
    out["prediction_id"] = out.apply(
        lambda r: hashlib.sha256(
            f"{OUTPUT_MODEL_VERSION}|{r['season']}|{r['as_of_date']}|{r['team_a_key']}|{r['team_b_key']}".encode("utf-8")
        ).hexdigest(),
        axis=1,
    )

    return out


def persist_predictions(
    con: duckdb.DuckDBPyConnection,
    blended_df: pd.DataFrame,
) -> int:
    if blended_df.empty:
        return 0

    payload = list(
        blended_df[
            [
                "prediction_id",
                "model_version",
                "season",
                "as_of_date",
                "team_a_key",
                "team_b_key",
                "p_model_a",
                "p_market_a",
                "p_blend_a",
                "calibration_bucket",
            ]
        ].itertuples(index=False, name=None)
    )

    con.executemany(
        """
        insert or replace into marts.matchup_prediction_fact
        (prediction_id, model_version, season, as_of_date, team_a_key, team_b_key,
         p_model_a, p_market_a, p_blend_a, calibration_bucket, created_at)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
        """,
        payload,
    )

    return len(payload)


def summarize(blended_df: pd.DataFrame) -> None:
    total = len(blended_df)
    with_market = int(blended_df["p_market_a"].notna().sum())
    without_market = total - with_market

    print(f"Rows written: {total}")
    print(f"Rows with market probabilities: {with_market}")
    print(f"Rows without market probabilities: {without_market}")
    print(f"Blend method: {BLEND_METHOD}")
    if BLEND_METHOD == "convex":
        print(f"Convex alpha (market weight): {BLEND_ALPHA:.3f}")
    else:
        print(f"Logit market weight: {BLEND_MARKET_WEIGHT:.3f}")
        print(f"Logit model weight: {BLEND_MODEL_WEIGHT:.3f}")
        print(f"Logit intercept: {BLEND_INTERCEPT:.3f}")

    preview = blended_df.sort_values("p_blend_a", ascending=False).head(12)[
        ["team_a_key", "team_b_key", "p_model_a", "p_market_a", "p_blend_a", "n_books"]
    ]
    print("\nSample blended predictions:")
    print(preview.to_string(index=False))


def main() -> int:
    con = connect_db()

    as_of_date = get_latest_prediction_date(
        con=con,
        season=SEASON,
        model_version=INPUT_MODEL_VERSION,
    )
    pred_df = load_input_predictions(
        con=con,
        season=SEASON,
        model_version=INPUT_MODEL_VERSION,
        as_of_date=as_of_date,
    )

    market_df = load_latest_market_rows(con=con, season=SEASON)
    if market_df.empty:
        print("No usable h2h market rows found in staging.market_odds_fact.")
        print("Run src.ingest.ingest_odds after your ODDS_API_KEY arrives, then rerun this file.")
        return 0

    market_dir_df = build_directional_market_probs(market_df)
    blended_df = apply_blend(pred_df=pred_df, market_dir_df=market_dir_df)
    n = persist_predictions(con=con, blended_df=blended_df)

    print(f"Input model version:  {INPUT_MODEL_VERSION}")
    print(f"Output model version: {OUTPUT_MODEL_VERSION}")
    print(f"As-of date:           {as_of_date}")
    print(f"Inserted/updated:     {n}")
    summarize(blended_df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())