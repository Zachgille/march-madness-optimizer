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

MATCHUP_AS_OF_DATE = os.getenv("MATCHUP_AS_OF_DATE", "").strip()
MATCHUP_MODEL_VERSION = os.getenv(
    "MATCHUP_MODEL_VERSION",
    os.getenv("OUTPUT_MODEL_VERSION", "logistic_seed_margin_v1_blend_v1"),
).strip()


def connect_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("create schema if not exists staging;")
    con.execute("create schema if not exists marts;")

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
        create table if not exists marts.matchup_feature_fact (
            matchup_key varchar primary key,
            season integer not null,
            as_of_date date not null,
            round_num integer,
            team_a_key varchar not null,
            team_b_key varchar not null,
            seed_diff integer,
            adj_off_diff double,
            adj_def_diff double,
            net_rating_diff double,
            tempo_diff double,
            sos_diff double,
            last_10_win_pct_diff double,
            last_5_net_rating_diff double,
            neutral_net_rating_diff double,
            efg_diff double,
            tov_rate_diff double,
            orb_rate_diff double,
            ftr_diff double,
            opp_efg_diff double,
            opp_tov_rate_diff double,
            opp_orb_rate_diff double,
            opp_ftr_diff double,
            market_prob_a double,
            market_prob_b double,
            market_spread_a double,
            market_total double
        );
        """
    )

    return con


def get_as_of_date(con: duckdb.DuckDBPyConnection) -> str:
    if MATCHUP_AS_OF_DATE:
        return str(pd.to_datetime(MATCHUP_AS_OF_DATE).date())

    row = con.execute(
        """
        select cast(max(as_of_date) as varchar)
        from staging.team_season_snapshot_fact
        where season = ?
        """,
        [SEASON],
    ).fetchone()

    if row is None or row[0] is None:
        raise RuntimeError(
            "No team_season_snapshot_fact rows found. Run build_team_game_features.py first."
        )

    return str(row[0])


def load_team_snapshot(con: duckdb.DuckDBPyConnection, as_of_date: str) -> pd.DataFrame:
    df = con.execute(
        """
        select
            season,
            as_of_date,
            team_key,
            games_played,
            wins,
            losses,
            net_rating,
            adj_off,
            adj_def,
            tempo,
            sos,
            last_10_win_pct,
            last_5_net_rating,
            neutral_net_rating,
            top_50_win_count,
            top_100_win_count,
            injury_adj,
            coach_adj
        from staging.team_season_snapshot_fact
        where season = ?
          and as_of_date = ?
        """,
        [SEASON, as_of_date],
    ).df()

    if df.empty:
        raise RuntimeError(
            f"No team snapshot rows found for season={SEASON}, as_of_date={as_of_date}."
        )

    return df


def load_field(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        """
        select
            season,
            team_key,
            region,
            seed,
            seed_label,
            play_in_group,
            is_play_in,
            initial_slot
        from staging.tournament_seed_dim
        where season = ?
        """,
        [SEASON],
    ).df()

    if df.empty:
        raise RuntimeError("No tournament_seed_dim rows found.")
    return df


def load_latest_market_rows(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        """
        select
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
          and team1_key is not null
          and team2_key is not null
        """,
        [SEASON],
    ).df()

    if df.empty:
        return df

    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    df["pair_low"] = df[["team1_key", "team2_key"]].min(axis=1)
    df["pair_high"] = df[["team1_key", "team2_key"]].max(axis=1)

    # Keep latest per bookmaker + unordered pair + market type
    df = (
        df.sort_values("snapshot_ts")
        .groupby(["bookmaker", "market_type", "pair_low", "pair_high"], as_index=False)
        .tail(1)
        .copy()
    )

    return df


def build_directional_market_features(market_df: pd.DataFrame) -> pd.DataFrame:
    if market_df.empty:
        return pd.DataFrame(
            columns=[
                "team_a_key",
                "team_b_key",
                "market_prob_a",
                "market_prob_b",
                "market_spread_a",
                "market_total",
            ]
        )

    h2h = market_df[market_df["market_type"] == "h2h"].copy()
    spreads = market_df[market_df["market_type"] == "spreads"].copy()
    totals = market_df[market_df["market_type"] == "totals"].copy()

    h2h_dir = pd.concat(
        [
            h2h[["team1_key", "team2_key", "implied_prob_team1_novig"]].rename(
                columns={
                    "team1_key": "team_a_key",
                    "team2_key": "team_b_key",
                    "implied_prob_team1_novig": "market_prob_a",
                }
            ),
            h2h[["team2_key", "team1_key", "implied_prob_team2_novig"]].rename(
                columns={
                    "team2_key": "team_a_key",
                    "team1_key": "team_b_key",
                    "implied_prob_team2_novig": "market_prob_a",
                }
            ),
        ],
        ignore_index=True,
    )

    spread_dir = pd.concat(
        [
            spreads[["team1_key", "team2_key", "spread_team1"]].rename(
                columns={
                    "team1_key": "team_a_key",
                    "team2_key": "team_b_key",
                    "spread_team1": "market_spread_a",
                }
            ),
            spreads[["team2_key", "team1_key", "spread_team2"]].rename(
                columns={
                    "team2_key": "team_a_key",
                    "team1_key": "team_b_key",
                    "spread_team2": "market_spread_a",
                }
            ),
        ],
        ignore_index=True,
    )

    total_dir = pd.concat(
        [
            totals[["team1_key", "team2_key", "total_points"]].rename(
                columns={
                    "team1_key": "team_a_key",
                    "team2_key": "team_b_key",
                    "total_points": "market_total",
                }
            ),
            totals[["team2_key", "team1_key", "total_points"]].rename(
                columns={
                    "team2_key": "team_a_key",
                    "team1_key": "team_b_key",
                    "total_points": "market_total",
                }
            ),
        ],
        ignore_index=True,
    )

    out = h2h_dir.groupby(["team_a_key", "team_b_key"], as_index=False).agg(
        market_prob_a=("market_prob_a", "mean")
    )
    out["market_prob_b"] = 1.0 - out["market_prob_a"]

    spread_agg = spread_dir.groupby(["team_a_key", "team_b_key"], as_index=False).agg(
        market_spread_a=("market_spread_a", "mean")
    )
    total_agg = total_dir.groupby(["team_a_key", "team_b_key"], as_index=False).agg(
        market_total=("market_total", "mean")
    )

    out = out.merge(spread_agg, on=["team_a_key", "team_b_key"], how="left")
    out = out.merge(total_agg, on=["team_a_key", "team_b_key"], how="left")
    return out


def build_team_pairs(field_df: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(field_df["team_key"].astype(str).unique().tolist())

    rows = []
    for team_a in teams:
        for team_b in teams:
            if team_a == team_b:
                continue
            rows.append({"team_a_key": team_a, "team_b_key": team_b})

    return pd.DataFrame(rows)


def safe_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")


def build_matchup_features(
    team_pairs: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    field_df: pd.DataFrame,
    market_features_df: pd.DataFrame,
    as_of_date: str,
) -> pd.DataFrame:
    seed_df = field_df[["team_key", "seed", "region"]].drop_duplicates().copy()

    a_stats = snapshot_df.rename(
        columns={
            "team_key": "team_a_key",
            "net_rating": "a_net_rating",
            "adj_off": "a_adj_off",
            "adj_def": "a_adj_def",
            "tempo": "a_tempo",
            "sos": "a_sos",
            "last_10_win_pct": "a_last_10_win_pct",
            "last_5_net_rating": "a_last_5_net_rating",
            "neutral_net_rating": "a_neutral_net_rating",
            "injury_adj": "a_injury_adj",
            "coach_adj": "a_coach_adj",
        }
    )

    b_stats = snapshot_df.rename(
        columns={
            "team_key": "team_b_key",
            "net_rating": "b_net_rating",
            "adj_off": "b_adj_off",
            "adj_def": "b_adj_def",
            "tempo": "b_tempo",
            "sos": "b_sos",
            "last_10_win_pct": "b_last_10_win_pct",
            "last_5_net_rating": "b_last_5_net_rating",
            "neutral_net_rating": "b_neutral_net_rating",
            "injury_adj": "b_injury_adj",
            "coach_adj": "b_coach_adj",
        }
    )

    a_seed = seed_df.rename(
        columns={
            "team_key": "team_a_key",
            "seed": "a_seed",
            "region": "a_region",
        }
    )

    b_seed = seed_df.rename(
        columns={
            "team_key": "team_b_key",
            "seed": "b_seed",
            "region": "b_region",
        }
    )

    out = team_pairs.merge(a_stats, on="team_a_key", how="left")
    out = out.merge(b_stats, on="team_b_key", how="left")
    out = out.merge(a_seed, on="team_a_key", how="left")
    out = out.merge(b_seed, on="team_b_key", how="left")
    out = out.merge(market_features_df, on=["team_a_key", "team_b_key"], how="left")

    out["season"] = SEASON
    out["as_of_date"] = pd.to_datetime(as_of_date).date()
    out["round_num"] = None

    out["seed_diff"] = safe_diff(out["a_seed"], out["b_seed"])
    out["adj_off_diff"] = safe_diff(out["a_adj_off"], out["b_adj_off"])
    out["adj_def_diff"] = safe_diff(out["a_adj_def"], out["b_adj_def"])
    out["net_rating_diff"] = safe_diff(
        out["a_net_rating"] + out["a_coach_adj"] + out["a_injury_adj"],
        out["b_net_rating"] + out["b_coach_adj"] + out["b_injury_adj"],
    )
    out["tempo_diff"] = safe_diff(out["a_tempo"], out["b_tempo"])
    out["sos_diff"] = safe_diff(out["a_sos"], out["b_sos"])
    out["last_10_win_pct_diff"] = safe_diff(out["a_last_10_win_pct"], out["b_last_10_win_pct"])
    out["last_5_net_rating_diff"] = safe_diff(out["a_last_5_net_rating"], out["b_last_5_net_rating"])
    out["neutral_net_rating_diff"] = safe_diff(out["a_neutral_net_rating"], out["b_neutral_net_rating"])

    # These are placeholders until team-game-level four-factor season summaries
    # are stored in a dedicated table. Keep them null-safe and present in schema.
    out["efg_diff"] = np.nan
    out["tov_rate_diff"] = np.nan
    out["orb_rate_diff"] = np.nan
    out["ftr_diff"] = np.nan
    out["opp_efg_diff"] = np.nan
    out["opp_tov_rate_diff"] = np.nan
    out["opp_orb_rate_diff"] = np.nan
    out["opp_ftr_diff"] = np.nan

    out["matchup_key"] = out.apply(
        lambda r: hashlib.sha256(
            f"{SEASON}|{as_of_date}|{r['team_a_key']}|{r['team_b_key']}".encode("utf-8")
        ).hexdigest(),
        axis=1,
    )

    cols = [
        "matchup_key",
        "season",
        "as_of_date",
        "round_num",
        "team_a_key",
        "team_b_key",
        "seed_diff",
        "adj_off_diff",
        "adj_def_diff",
        "net_rating_diff",
        "tempo_diff",
        "sos_diff",
        "last_10_win_pct_diff",
        "last_5_net_rating_diff",
        "neutral_net_rating_diff",
        "efg_diff",
        "tov_rate_diff",
        "orb_rate_diff",
        "ftr_diff",
        "opp_efg_diff",
        "opp_tov_rate_diff",
        "opp_orb_rate_diff",
        "opp_ftr_diff",
        "market_prob_a",
        "market_prob_b",
        "market_spread_a",
        "market_total",
    ]

    return out[cols].copy()


def persist_matchup_features(
    con: duckdb.DuckDBPyConnection,
    matchup_df: pd.DataFrame,
    as_of_date: str,
) -> int:
    con.execute(
        """
        delete from marts.matchup_feature_fact
        where season = ?
          and as_of_date = ?
        """,
        [SEASON, as_of_date],
    )

    payload = list(
        matchup_df[
            [
                "matchup_key",
                "season",
                "as_of_date",
                "round_num",
                "team_a_key",
                "team_b_key",
                "seed_diff",
                "adj_off_diff",
                "adj_def_diff",
                "net_rating_diff",
                "tempo_diff",
                "sos_diff",
                "last_10_win_pct_diff",
                "last_5_net_rating_diff",
                "neutral_net_rating_diff",
                "efg_diff",
                "tov_rate_diff",
                "orb_rate_diff",
                "ftr_diff",
                "opp_efg_diff",
                "opp_tov_rate_diff",
                "opp_orb_rate_diff",
                "opp_ftr_diff",
                "market_prob_a",
                "market_prob_b",
                "market_spread_a",
                "market_total",
            ]
        ].itertuples(index=False, name=None)
    )

    con.executemany(
        """
        insert into marts.matchup_feature_fact
        (matchup_key, season, as_of_date, round_num, team_a_key, team_b_key,
         seed_diff, adj_off_diff, adj_def_diff, net_rating_diff, tempo_diff, sos_diff,
         last_10_win_pct_diff, last_5_net_rating_diff, neutral_net_rating_diff,
         efg_diff, tov_rate_diff, orb_rate_diff, ftr_diff,
         opp_efg_diff, opp_tov_rate_diff, opp_orb_rate_diff, opp_ftr_diff,
         market_prob_a, market_prob_b, market_spread_a, market_total)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )

    return len(payload)


def main() -> int:
    con = connect_db()
    as_of_date = get_as_of_date(con)

    snapshot_df = load_team_snapshot(con, as_of_date)
    field_df = load_field(con)
    market_df = load_latest_market_rows(con)
    market_features_df = build_directional_market_features(market_df)
    team_pairs = build_team_pairs(field_df)

    matchup_df = build_matchup_features(
        team_pairs=team_pairs,
        snapshot_df=snapshot_df,
        field_df=field_df,
        market_features_df=market_features_df,
        as_of_date=as_of_date,
    )

    n = persist_matchup_features(con, matchup_df, as_of_date)

    print(f"Season:       {SEASON}")
    print(f"As-of date:   {as_of_date}")
    print(f"Model hint:   {MATCHUP_MODEL_VERSION}")
    print(f"Field teams:  {field_df['team_key'].nunique()}")
    print(f"Matchup rows: {n}")
    print(f"With market:  {int(matchup_df['market_prob_a'].notna().sum())}")

    preview = matchup_df.sort_values(
        ["market_prob_a", "net_rating_diff"],
        ascending=[False, False],
        na_position="last",
    ).head(15)
    print("\nSample matchup features:")
    print(
        preview[
            [
                "team_a_key",
                "team_b_key",
                "seed_diff",
                "net_rating_diff",
                "market_prob_a",
                "market_spread_a",
                "market_total",
            ]
        ].to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())