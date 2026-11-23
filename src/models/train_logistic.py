from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

load_dotenv()

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser()
DUCKDB_PATH = Path(
    os.getenv("DUCKDB_PATH", str(PROJECT_ROOT / "data" / "march_madness.duckdb"))
).expanduser()
SEASON = int(os.getenv("SEASON", "2026"))

KAGGLE_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kaggle"
MODEL_OUT_DIR = PROJECT_ROOT / "models"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = os.getenv("INPUT_MODEL_VERSION", "logistic_matchup_v2").strip()
MATCHUP_AS_OF_DATE = os.getenv("MATCHUP_AS_OF_DATE", "").strip()

MASTER_FEATURE_COLS = [
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
]


@dataclass
class FitArtifacts:
    pipeline: Pipeline
    metrics: dict
    feature_cols: list[str]
    train_rows: int
    test_rows: int


def connect_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("create schema if not exists marts;")
    con.execute("create schema if not exists staging;")

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


def find_first_existing(candidates: Iterable[str]) -> Path:
    for name in candidates:
        path = KAGGLE_RAW_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files were found in {KAGGLE_RAW_DIR}: {list(candidates)}")


def maybe_find_first_existing(candidates: Iterable[str]) -> Path | None:
    for name in candidates:
        path = KAGGLE_RAW_DIR / name
        if path.exists():
            return path
    return None


def extract_seed_num(seed_value: str) -> int:
    s = str(seed_value)
    m = re.search(r"(\d{1,2})", s)
    if not m:
        raise ValueError(f"Could not parse numeric seed from {seed_value}")
    return int(m.group(1))


def load_kaggle_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    reg_compact = pd.read_csv(find_first_existing(["MRegularSeasonCompactResults.csv"]))
    tourney_compact = pd.read_csv(find_first_existing(["MNCAATourneyCompactResults.csv"]))
    seeds = pd.read_csv(find_first_existing(["MNCAATourneySeeds.csv"]))

    reg_detailed_path = maybe_find_first_existing(["MRegularSeasonDetailedResults.csv"])
    reg_detailed = pd.read_csv(reg_detailed_path) if reg_detailed_path else None

    return reg_compact, tourney_compact, seeds, reg_detailed


def build_team_rows_from_detailed(reg_detailed: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(
        {
            "Season": reg_detailed["Season"],
            "DayNum": reg_detailed["DayNum"],
            "TeamID": reg_detailed["WTeamID"],
            "OppTeamID": reg_detailed["LTeamID"],
            "PointsFor": reg_detailed["WScore"],
            "PointsAgainst": reg_detailed["LScore"],
            "FGM": reg_detailed["WFGM"],
            "FGA": reg_detailed["WFGA"],
            "FG3M": reg_detailed["WFGM3"],
            "FG3A": reg_detailed["WFGA3"],
            "FTM": reg_detailed["WFTM"],
            "FTA": reg_detailed["WFTA"],
            "ORB": reg_detailed["WOR"],
            "DRB": reg_detailed["WDR"],
            "AST": reg_detailed["WAst"],
            "TO": reg_detailed["WTO"],
            "OppFGM": reg_detailed["LFGM"],
            "OppFGA": reg_detailed["LFGA"],
            "OppFG3M": reg_detailed["LFGM3"],
            "OppFG3A": reg_detailed["LFGA3"],
            "OppFTM": reg_detailed["LFTM"],
            "OppFTA": reg_detailed["LFTA"],
            "OppORB": reg_detailed["LOR"],
            "OppDRB": reg_detailed["LDR"],
            "OppAST": reg_detailed["LAst"],
            "OppTO": reg_detailed["LTO"],
            "IsWin": 1,
            "IsNeutral": (reg_detailed["WLoc"] == "N").astype(int),
        }
    )

    l = pd.DataFrame(
        {
            "Season": reg_detailed["Season"],
            "DayNum": reg_detailed["DayNum"],
            "TeamID": reg_detailed["LTeamID"],
            "OppTeamID": reg_detailed["WTeamID"],
            "PointsFor": reg_detailed["LScore"],
            "PointsAgainst": reg_detailed["WScore"],
            "FGM": reg_detailed["LFGM"],
            "FGA": reg_detailed["LFGA"],
            "FG3M": reg_detailed["LFGM3"],
            "FG3A": reg_detailed["LFGA3"],
            "FTM": reg_detailed["LFTM"],
            "FTA": reg_detailed["LFTA"],
            "ORB": reg_detailed["LOR"],
            "DRB": reg_detailed["LDR"],
            "AST": reg_detailed["LAst"],
            "TO": reg_detailed["LTO"],
            "OppFGM": reg_detailed["WFGM"],
            "OppFGA": reg_detailed["WFGA"],
            "OppFG3M": reg_detailed["WFGM3"],
            "OppFG3A": reg_detailed["WFGA3"],
            "OppFTM": reg_detailed["WFTM"],
            "OppFTA": reg_detailed["WFTA"],
            "OppORB": reg_detailed["WOR"],
            "OppDRB": reg_detailed["WDR"],
            "OppAST": reg_detailed["WAst"],
            "OppTO": reg_detailed["WTO"],
            "IsWin": 0,
            "IsNeutral": (reg_detailed["WLoc"] == "N").astype(int),
        }
    )

    games = pd.concat([w, l], ignore_index=True)

    games["Poss"] = games["FGA"] - games["ORB"] + games["TO"] + 0.475 * games["FTA"]
    games["OppPoss"] = games["OppFGA"] - games["OppORB"] + games["OppTO"] + 0.475 * games["OppFTA"]
    games["PossAvg"] = (games["Poss"] + games["OppPoss"]) / 2.0

    games["ORtg"] = 100.0 * games["PointsFor"] / games["PossAvg"].clip(lower=1e-9)
    games["DRtg"] = 100.0 * games["PointsAgainst"] / games["PossAvg"].clip(lower=1e-9)
    games["NetGameRating"] = games["ORtg"] - games["DRtg"]

    games["eFG"] = (games["FGM"] + 0.5 * games["FG3M"]) / games["FGA"].clip(lower=1e-9)
    games["TOVRate"] = games["TO"] / games["PossAvg"].clip(lower=1e-9)
    games["ORBRate"] = games["ORB"] / (games["ORB"] + games["OppDRB"]).clip(lower=1e-9)
    games["FTR"] = games["FTA"] / games["FGA"].clip(lower=1e-9)

    games["Opp_eFG"] = (games["OppFGM"] + 0.5 * games["OppFG3M"]) / games["OppFGA"].clip(lower=1e-9)
    games["Opp_TOVRate"] = games["OppTO"] / games["OppPoss"].clip(lower=1e-9)
    games["Opp_ORBRate"] = games["OppORB"] / (games["OppORB"] + games["DRB"]).clip(lower=1e-9)
    games["Opp_FTR"] = games["OppFTA"] / games["OppFGA"].clip(lower=1e-9)

    return games


def build_team_rows_from_compact(reg_compact: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(
        {
            "Season": reg_compact["Season"],
            "DayNum": reg_compact["DayNum"],
            "TeamID": reg_compact["WTeamID"],
            "OppTeamID": reg_compact["LTeamID"],
            "PointsFor": reg_compact["WScore"],
            "PointsAgainst": reg_compact["LScore"],
            "IsWin": 1,
            "IsNeutral": (reg_compact["WLoc"] == "N").astype(int),
        }
    )

    l = pd.DataFrame(
        {
            "Season": reg_compact["Season"],
            "DayNum": reg_compact["DayNum"],
            "TeamID": reg_compact["LTeamID"],
            "OppTeamID": reg_compact["WTeamID"],
            "PointsFor": reg_compact["LScore"],
            "PointsAgainst": reg_compact["WScore"],
            "IsWin": 0,
            "IsNeutral": (reg_compact["WLoc"] == "N").astype(int),
        }
    )

    games = pd.concat([w, l], ignore_index=True)
    games["NetGameRating"] = games["PointsFor"] - games["PointsAgainst"]
    return games


def build_historical_team_snapshot(
    reg_compact: pd.DataFrame,
    reg_detailed: pd.DataFrame | None,
) -> pd.DataFrame:
    if reg_detailed is not None:
        games = build_team_rows_from_detailed(reg_detailed)

        grouped = (
            games.groupby(["Season", "TeamID"], as_index=False)
            .agg(
                games_played=("IsWin", "count"),
                wins=("IsWin", "sum"),
                points_for=("PointsFor", "sum"),
                points_against=("PointsAgainst", "sum"),
                poss=("PossAvg", "sum"),
                tempo=("PossAvg", "mean"),
                efg=("eFG", "mean"),
                tov_rate=("TOVRate", "mean"),
                orb_rate=("ORBRate", "mean"),
                ftr=("FTR", "mean"),
                opp_efg=("Opp_eFG", "mean"),
                opp_tov_rate=("Opp_TOVRate", "mean"),
                opp_orb_rate=("Opp_ORBRate", "mean"),
                opp_ftr=("Opp_FTR", "mean"),
            )
        )

        grouped["losses"] = grouped["games_played"] - grouped["wins"]
        grouped["adj_off"] = 100.0 * grouped["points_for"] / grouped["poss"].clip(lower=1e-9)
        grouped["adj_def"] = 100.0 * grouped["points_against"] / grouped["poss"].clip(lower=1e-9)
        grouped["net_rating"] = grouped["adj_off"] - grouped["adj_def"]
    else:
        games = build_team_rows_from_compact(reg_compact)

        grouped = (
            games.groupby(["Season", "TeamID"], as_index=False)
            .agg(
                games_played=("IsWin", "count"),
                wins=("IsWin", "sum"),
                points_for=("PointsFor", "mean"),
                points_against=("PointsAgainst", "mean"),
            )
        )

        grouped["losses"] = grouped["games_played"] - grouped["wins"]
        grouped["adj_off"] = grouped["points_for"]
        grouped["adj_def"] = grouped["points_against"]
        grouped["net_rating"] = grouped["points_for"] - grouped["points_against"]
        grouped["tempo"] = np.nan
        grouped["efg"] = np.nan
        grouped["tov_rate"] = np.nan
        grouped["orb_rate"] = np.nan
        grouped["ftr"] = np.nan
        grouped["opp_efg"] = np.nan
        grouped["opp_tov_rate"] = np.nan
        grouped["opp_orb_rate"] = np.nan
        grouped["opp_ftr"] = np.nan

    grouped["win_pct"] = grouped["wins"] / grouped["games_played"].clip(lower=1)

    last10 = (
        games.sort_values(["Season", "TeamID", "DayNum"])
        .groupby(["Season", "TeamID"])
        .tail(10)
        .groupby(["Season", "TeamID"], as_index=False)
        .agg(last_10_win_pct=("IsWin", "mean"))
    )

    last5 = (
        games.sort_values(["Season", "TeamID", "DayNum"])
        .groupby(["Season", "TeamID"])
        .tail(5)
        .groupby(["Season", "TeamID"], as_index=False)
        .agg(last_5_net_rating=("NetGameRating", "mean"))
    )

    neutral = (
        games[games["IsNeutral"] == 1]
        .groupby(["Season", "TeamID"], as_index=False)
        .agg(neutral_net_rating=("NetGameRating", "mean"))
    )

    grouped = grouped.merge(last10, on=["Season", "TeamID"], how="left")
    grouped = grouped.merge(last5, on=["Season", "TeamID"], how="left")
    grouped = grouped.merge(neutral, on=["Season", "TeamID"], how="left")

    grouped["last_10_win_pct"] = grouped["last_10_win_pct"].fillna(grouped["win_pct"])
    grouped["last_5_net_rating"] = grouped["last_5_net_rating"].fillna(grouped["net_rating"])
    grouped["neutral_net_rating"] = grouped["neutral_net_rating"].fillna(grouped["net_rating"])

    base_strength = grouped[["Season", "TeamID", "net_rating"]].rename(
        columns={"TeamID": "OppTeamID", "net_rating": "opp_strength"}
    )
    enriched = games.merge(base_strength, on=["Season", "OppTeamID"], how="left")
    sos = (
        enriched.groupby(["Season", "TeamID"], as_index=False)
        .agg(sos=("opp_strength", "mean"))
    )
    grouped = grouped.merge(sos, on=["Season", "TeamID"], how="left")
    grouped["sos"] = grouped["sos"].fillna(0.0)

    return grouped[
        [
            "Season",
            "TeamID",
            "games_played",
            "wins",
            "losses",
            "adj_off",
            "adj_def",
            "net_rating",
            "tempo",
            "sos",
            "last_10_win_pct",
            "last_5_net_rating",
            "neutral_net_rating",
            "efg",
            "tov_rate",
            "orb_rate",
            "ftr",
            "opp_efg",
            "opp_tov_rate",
            "opp_orb_rate",
            "opp_ftr",
        ]
    ]


def build_seeds_numeric(seeds: pd.DataFrame) -> pd.DataFrame:
    out = seeds.copy()
    out["seed_num"] = out["Seed"].map(extract_seed_num)
    return out[["Season", "TeamID", "seed_num"]]


def build_training_rows(
    team_snapshot: pd.DataFrame,
    tourney_compact: pd.DataFrame,
    seeds_numeric: pd.DataFrame,
) -> pd.DataFrame:
    winners = tourney_compact.rename(
        columns={"WTeamID": "TeamAID", "LTeamID": "TeamBID"}
    )[["Season", "TeamAID", "TeamBID"]].copy()
    winners["y"] = 1

    losers = tourney_compact.rename(
        columns={"LTeamID": "TeamAID", "WTeamID": "TeamBID"}
    )[["Season", "TeamAID", "TeamBID"]].copy()
    losers["y"] = 0

    train = pd.concat([winners, losers], ignore_index=True)

    a_stats = team_snapshot.rename(
        columns={
            "TeamID": "TeamAID",
            "adj_off": "a_adj_off",
            "adj_def": "a_adj_def",
            "net_rating": "a_net_rating",
            "tempo": "a_tempo",
            "sos": "a_sos",
            "last_10_win_pct": "a_last_10_win_pct",
            "last_5_net_rating": "a_last_5_net_rating",
            "neutral_net_rating": "a_neutral_net_rating",
            "efg": "a_efg",
            "tov_rate": "a_tov_rate",
            "orb_rate": "a_orb_rate",
            "ftr": "a_ftr",
            "opp_efg": "a_opp_efg",
            "opp_tov_rate": "a_opp_tov_rate",
            "opp_orb_rate": "a_opp_orb_rate",
            "opp_ftr": "a_opp_ftr",
        }
    )

    b_stats = team_snapshot.rename(
        columns={
            "TeamID": "TeamBID",
            "adj_off": "b_adj_off",
            "adj_def": "b_adj_def",
            "net_rating": "b_net_rating",
            "tempo": "b_tempo",
            "sos": "b_sos",
            "last_10_win_pct": "b_last_10_win_pct",
            "last_5_net_rating": "b_last_5_net_rating",
            "neutral_net_rating": "b_neutral_net_rating",
            "efg": "b_efg",
            "tov_rate": "b_tov_rate",
            "orb_rate": "b_orb_rate",
            "ftr": "b_ftr",
            "opp_efg": "b_opp_efg",
            "opp_tov_rate": "b_opp_tov_rate",
            "opp_orb_rate": "b_opp_orb_rate",
            "opp_ftr": "b_opp_ftr",
        }
    )

    a_seed = seeds_numeric.rename(columns={"TeamID": "TeamAID", "seed_num": "a_seed"})
    b_seed = seeds_numeric.rename(columns={"TeamID": "TeamBID", "seed_num": "b_seed"})

    train = train.merge(a_stats, on=["Season", "TeamAID"], how="left")
    train = train.merge(b_stats, on=["Season", "TeamBID"], how="left")
    train = train.merge(a_seed, on=["Season", "TeamAID"], how="left")
    train = train.merge(b_seed, on=["Season", "TeamBID"], how="left")

    train["seed_diff"] = train["a_seed"] - train["b_seed"]
    train["adj_off_diff"] = train["a_adj_off"] - train["b_adj_off"]
    train["adj_def_diff"] = train["a_adj_def"] - train["b_adj_def"]
    train["net_rating_diff"] = train["a_net_rating"] - train["b_net_rating"]
    train["tempo_diff"] = train["a_tempo"] - train["b_tempo"]
    train["sos_diff"] = train["a_sos"] - train["b_sos"]
    train["last_10_win_pct_diff"] = train["a_last_10_win_pct"] - train["b_last_10_win_pct"]
    train["last_5_net_rating_diff"] = train["a_last_5_net_rating"] - train["b_last_5_net_rating"]
    train["neutral_net_rating_diff"] = train["a_neutral_net_rating"] - train["b_neutral_net_rating"]
    train["efg_diff"] = train["a_efg"] - train["b_efg"]
    train["tov_rate_diff"] = train["a_tov_rate"] - train["b_tov_rate"]
    train["orb_rate_diff"] = train["a_orb_rate"] - train["b_orb_rate"]
    train["ftr_diff"] = train["a_ftr"] - train["b_ftr"]
    train["opp_efg_diff"] = train["a_opp_efg"] - train["b_opp_efg"]
    train["opp_tov_rate_diff"] = train["a_opp_tov_rate"] - train["b_opp_tov_rate"]
    train["opp_orb_rate_diff"] = train["a_opp_orb_rate"] - train["b_opp_orb_rate"]
    train["opp_ftr_diff"] = train["a_opp_ftr"] - train["b_opp_ftr"]

    train = train.dropna(subset=["y"]).copy()
    return train


def choose_feature_cols(train_df: pd.DataFrame) -> list[str]:
    feature_cols: list[str] = []
    for col in MASTER_FEATURE_COLS:
        if col not in train_df.columns:
            continue
        non_null = train_df[col].notna().sum()
        nunique = train_df[col].nunique(dropna=True)
        if non_null == 0:
            continue
        if nunique <= 1:
            continue
        feature_cols.append(col)

    if not feature_cols:
        raise RuntimeError("No usable training features were constructed.")
    return feature_cols


def fit_model(train_df: pd.DataFrame, feature_cols: list[str]) -> FitArtifacts:
    seasons = sorted(train_df["Season"].unique().tolist())
    if len(seasons) < 4:
        raise RuntimeError("Not enough historical seasons found to fit/evaluate reliably.")

    holdout_seasons = seasons[-5:] if len(seasons) >= 8 else seasons[-2:]

    train_part = train_df[~train_df["Season"].isin(holdout_seasons)].copy()
    test_part = train_df[train_df["Season"].isin(holdout_seasons)].copy()

    X_train = train_part[feature_cols]
    y_train = train_part["y"].astype(int)

    X_test = test_part[feature_cols]
    y_test = test_part["y"].astype(int)

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")),
        ]
    )
    pipe.fit(X_train, y_train)

    p_test = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "model_version": MODEL_VERSION,
        "holdout_seasons": holdout_seasons,
        "log_loss": float(log_loss(y_test, p_test)),
        "brier_score": float(brier_score_loss(y_test, p_test)),
    }

    return FitArtifacts(
        pipeline=pipe,
        metrics=metrics,
        feature_cols=feature_cols,
        train_rows=len(train_part),
        test_rows=len(test_part),
    )


def save_artifacts(artifacts: FitArtifacts) -> tuple[Path, Path]:
    model_path = MODEL_OUT_DIR / f"{MODEL_VERSION}.pkl"
    metrics_path = MODEL_OUT_DIR / f"{MODEL_VERSION}_metrics.json"

    clf = artifacts.pipeline.named_steps["clf"]
    coefs = dict(zip(artifacts.feature_cols, clf.coef_[0].tolist()))

    with model_path.open("wb") as f:
        pickle.dump(
            {
                "model_version": MODEL_VERSION,
                "feature_cols": artifacts.feature_cols,
                "pipeline": artifacts.pipeline,
            },
            f,
        )

    payload = {
        **artifacts.metrics,
        "train_rows": artifacts.train_rows,
        "test_rows": artifacts.test_rows,
        "feature_cols": artifacts.feature_cols,
        "coefficients": coefs,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return model_path, metrics_path


def get_latest_matchup_as_of_date(con: duckdb.DuckDBPyConnection) -> str:
    if MATCHUP_AS_OF_DATE:
        return str(pd.to_datetime(MATCHUP_AS_OF_DATE).date())

    row = con.execute(
        """
        select cast(max(as_of_date) as varchar)
        from marts.matchup_feature_fact
        where season = ?
        """,
        [SEASON],
    ).fetchone()

    if row is None or row[0] is None:
        raise RuntimeError(
            "No marts.matchup_feature_fact rows found. Run build_matchup_features.py first."
        )
    return str(row[0])


def load_current_matchup_features(con: duckdb.DuckDBPyConnection, as_of_date: str) -> pd.DataFrame:
    df = con.execute(
        """
        select
            team_a_key,
            team_b_key,
            market_prob_a,
            market_prob_b,
            market_spread_a,
            market_total,
            seed_diff,
            adj_off_diff,
            adj_def_diff,
            net_rating_diff,
            tempo_diff,
            sos_diff,
            last_10_win_pct_diff,
            last_5_net_rating_diff,
            neutral_net_rating_diff,
            efg_diff,
            tov_rate_diff,
            orb_rate_diff,
            ftr_diff,
            opp_efg_diff,
            opp_tov_rate_diff,
            opp_orb_rate_diff,
            opp_ftr_diff
        from marts.matchup_feature_fact
        where season = ?
          and as_of_date = ?
        """,
        [SEASON, as_of_date],
    ).df()

    if df.empty:
        raise RuntimeError(
            f"No matchup_feature_fact rows found for season={SEASON}, as_of_date={as_of_date}."
        )
    return df


def calibration_bucket(p: float) -> str:
    bucket = int(np.floor(max(0.0, min(0.999999, p)) * 20)) / 20
    return f"{bucket:.2f}-{bucket + 0.05:.2f}"


def score_current_field(
    con: duckdb.DuckDBPyConnection,
    artifacts: FitArtifacts,
) -> tuple[pd.DataFrame, str]:
    as_of_date = get_latest_matchup_as_of_date(con)
    matchup_df = load_current_matchup_features(con, as_of_date)

    X = matchup_df.copy()
    for col in artifacts.feature_cols:
        if col not in X.columns:
            X[col] = np.nan

    p_model = artifacts.pipeline.predict_proba(X[artifacts.feature_cols])[:, 1]

    pred_df = pd.DataFrame(
        {
            "prediction_id": [
                hashlib.sha256(
                    f"{MODEL_VERSION}|{SEASON}|{as_of_date}|{a}|{b}".encode("utf-8")
                ).hexdigest()
                for a, b in zip(matchup_df["team_a_key"], matchup_df["team_b_key"])
            ],
            "model_version": MODEL_VERSION,
            "season": SEASON,
            "as_of_date": pd.to_datetime(as_of_date).date(),
            "team_a_key": matchup_df["team_a_key"],
            "team_b_key": matchup_df["team_b_key"],
            "p_model_a": p_model,
            "p_market_a": pd.to_numeric(matchup_df["market_prob_a"], errors="coerce"),
            "p_blend_a": p_model,
            "calibration_bucket": [calibration_bucket(float(p)) for p in p_model],
        }
    )

    return pred_df, as_of_date


def persist_current_predictions(
    con: duckdb.DuckDBPyConnection,
    pred_df: pd.DataFrame,
) -> int:
    if pred_df.empty:
        return 0

    payload = list(
        pred_df[
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


def main() -> int:
    con = connect_db()

    reg_compact, tourney_compact, seeds, reg_detailed = load_kaggle_files()
    team_snapshot = build_historical_team_snapshot(reg_compact, reg_detailed)
    seeds_numeric = build_seeds_numeric(seeds)
    train_df = build_training_rows(team_snapshot, tourney_compact, seeds_numeric)
    feature_cols = choose_feature_cols(train_df)

    artifacts = fit_model(train_df, feature_cols)
    model_path, metrics_path = save_artifacts(artifacts)

    print(f"Model saved to:   {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Model version:    {MODEL_VERSION}")
    print(f"Features used:    {artifacts.feature_cols}")
    print(f"Train rows:       {artifacts.train_rows}")
    print(f"Test rows:        {artifacts.test_rows}")
    print("Holdout metrics:")
    for k, v in artifacts.metrics.items():
        print(f"  {k}: {v}")

    try:
        pred_df, as_of_date = score_current_field(con, artifacts)
        n = persist_current_predictions(con, pred_df)
        print(f"\nInserted current matchup predictions: {n}")
        print(f"Inference as_of_date: {as_of_date}")

        preview = pred_df.sort_values("p_model_a", ascending=False).head(12)[
            ["team_a_key", "team_b_key", "p_model_a", "p_market_a"]
        ]
        print("\nSample current predictions:")
        print(preview.to_string(index=False))
    except Exception as e:
        print(f"\nModel fit succeeded, but current-field scoring was skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())