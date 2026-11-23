from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

EVAL_MODEL_VERSION = os.getenv(
    "EVAL_MODEL_VERSION",
    os.getenv("CANDIDATE_MODEL_VERSION", os.getenv("SIM_MODEL_VERSION", "logistic_seed_margin_v1_blend_v1")),
).strip()

EVAL_PUBLIC_SOURCE = os.getenv("EVAL_PUBLIC_SOURCE", os.getenv("CANDIDATE_PUBLIC_SOURCE", "yahoo")).strip().lower()
EVAL_N = int(os.getenv("EVAL_N", "5000"))
EVAL_RANDOM_SEED = int(os.getenv("EVAL_RANDOM_SEED", "20260317"))

# Pool assumptions
# POOL_SIZE includes your entry.
POOL_SIZE = int(os.getenv("POOL_SIZE", "25"))

# Examples:
#   PAYOUTS=1:100
#   PAYOUTS=1:70,2:20,3:10
PAYOUTS_ENV = os.getenv("PAYOUTS", "1:100").strip()

# Opponent-generation weights
# Public pick share should dominate opponent behavior, but not override coherence.
OPP_PUBLIC_WEIGHT = float(os.getenv("OPP_PUBLIC_WEIGHT", "0.75"))
OPP_TRUE_WEIGHT = float(os.getenv("OPP_TRUE_WEIGHT", "0.25"))
OPP_NOISE_STD = float(os.getenv("OPP_NOISE_STD", "0.015"))

REGION_ORDER_ENV = os.getenv("REGION_ORDER", "").strip()
FINAL_FOUR_PAIRINGS_ENV = os.getenv("FINAL_FOUR_PAIRINGS", "").strip()

POINTS_BY_ROUND = {
    1: 10,
    2: 20,
    3: 40,
    4: 80,
    5: 160,
    6: 320,
}

OPENING_MATCHUPS = [
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
]


@dataclass
class OpponentTemplate:
    style: str  # public, balanced, chalky, contrarian


def connect_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("create schema if not exists staging;")
    con.execute("create schema if not exists marts;")

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

    con.execute(
        """
        create table if not exists marts.candidate_bracket (
            bracket_id varchar primary key,
            season integer not null,
            objective_type varchar not null,
            model_version varchar not null,
            created_at timestamp not null,
            picks_json varchar not null,
            champion_team_key varchar,
            finalist_1_team_key varchar,
            finalist_2_team_key varchar,
            expected_points double,
            first_place_prob double,
            top_3_prob double,
            expected_payout double,
            duplication_score double,
            notes varchar
        );
        """
    )

    con.execute(
        """
        create table if not exists marts.simulation_summary_fact (
            sim_run_id varchar not null,
            bracket_id varchar not null,
            season integer not null,
            n_sims integer not null,
            expected_points double,
            variance_points double,
            first_place_prob double,
            cash_prob double,
            expected_payout double,
            primary key (sim_run_id, bracket_id)
        );
        """
    )

    return con


def latest_prediction_date(con: duckdb.DuckDBPyConnection) -> str:
    row = con.execute(
        """
        select cast(max(as_of_date) as varchar)
        from marts.matchup_prediction_fact
        where season = ?
          and model_version = ?
        """,
        [SEASON, EVAL_MODEL_VERSION],
    ).fetchone()

    if row is None or row[0] is None:
        raise RuntimeError(
            f"No matchup_prediction_fact rows found for season={SEASON}, model_version={EVAL_MODEL_VERSION}."
        )
    return str(row[0])


def load_prediction_df(con: duckdb.DuckDBPyConnection, as_of_date: str) -> pd.DataFrame:
    df = con.execute(
        """
        select team_a_key, team_b_key, p_model_a, p_market_a, p_blend_a
        from marts.matchup_prediction_fact
        where season = ?
          and model_version = ?
          and as_of_date = ?
        """,
        [SEASON, EVAL_MODEL_VERSION, as_of_date],
    ).df()

    if df.empty:
        raise RuntimeError("No matchup predictions found.")
    return df


def load_field(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        """
        select season, team_key, region, seed, seed_label, play_in_group, is_play_in, initial_slot
        from staging.tournament_seed_dim
        where season = ?
        order by region, seed, seed_label, team_key
        """,
        [SEASON],
    ).df()

    if df.empty:
        raise RuntimeError("No tournament field found.")
    return df


def load_candidates(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        """
        select
            bracket_id,
            objective_type,
            model_version,
            picks_json,
            champion_team_key,
            finalist_1_team_key,
            finalist_2_team_key,
            expected_points,
            first_place_prob,
            top_3_prob,
            expected_payout,
            duplication_score,
            notes
        from marts.candidate_bracket
        where season = ?
          and model_version = ?
        order by expected_points desc nulls last, bracket_id
        """,
        [SEASON, EVAL_MODEL_VERSION],
    ).df()

    if df.empty:
        raise RuntimeError(
            f"No candidate_bracket rows found for season={SEASON}, model_version={EVAL_MODEL_VERSION}."
        )
    return df


def load_public_picks(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    expected_cols = [
        "team_key",
        "round_num",
        "pct_pick_to_advance",
        "pct_pick_to_reach_round",
        "pct_pick_to_win_title",
    ]

    try:
        df = con.execute(
            """
            select
                season,
                source_name,
                snapshot_ts,
                round_num,
                team_key,
                pct_pick_to_advance,
                pct_pick_to_reach_round,
                pct_pick_to_win_title
            from staging.public_pick_fact
            where season = ?
              and source_name = ?
            """,
            [SEASON, EVAL_PUBLIC_SOURCE],
        ).df()
    except Exception as e:
        print(f"[warn] Could not load public picks; continuing without them: {e}")
        return pd.DataFrame(columns=expected_cols)

    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    df = df.copy()
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce")
    df["round_num"] = pd.to_numeric(df["round_num"], errors="coerce")

    df = df.dropna(subset=["team_key", "round_num"]).copy()
    df["round_num"] = df["round_num"].astype(int)

    df = (
        df.sort_values("snapshot_ts")
        .drop_duplicates(subset=["team_key", "round_num"], keep="last")
        .copy()
    )

    return df[
        [
            "team_key",
            "round_num",
            "pct_pick_to_advance",
            "pct_pick_to_reach_round",
            "pct_pick_to_win_title",
        ]
    ]


def parse_region_order(field_df: pd.DataFrame) -> list[str]:
    regions = sorted(field_df["region"].dropna().astype(str).unique().tolist())

    if REGION_ORDER_ENV:
        requested = [x.strip() for x in REGION_ORDER_ENV.split(",") if x.strip()]
        if set(requested) != set(regions):
            raise ValueError(f"REGION_ORDER must match tournament regions exactly. Known regions: {regions}")
        return requested

    return regions


def parse_final_four_pairings(region_order: list[str]) -> list[tuple[str, str]]:
    if FINAL_FOUR_PAIRINGS_ENV:
        pairs: list[tuple[str, str]] = []
        used: list[str] = []
        for token in FINAL_FOUR_PAIRINGS_ENV.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" not in token:
                raise ValueError("FINAL_FOUR_PAIRINGS must look like 'East-West,South-Midwest'")
            a, b = [x.strip() for x in token.split("-", 1)]
            pairs.append((a, b))
            used.extend([a, b])
        if sorted(used) != sorted(region_order):
            raise ValueError("FINAL_FOUR_PAIRINGS must use each region exactly once.")
        return pairs

    if len(region_order) != 4:
        raise ValueError("Automatic semifinal pairing assumes exactly 4 regions.")
    return [(region_order[0], region_order[1]), (region_order[2], region_order[3])]


def build_region_seed_map(field_df: pd.DataFrame) -> dict[str, dict[int, list[str]]]:
    out: dict[str, dict[int, list[str]]] = {}
    for region, g in field_df.groupby("region"):
        seed_map: dict[int, list[str]] = {}
        for seed, sg in g.groupby("seed"):
            seed_map[int(seed)] = sorted(sg["team_key"].astype(str).tolist())
        out[str(region)] = seed_map
    return out


def build_prob_map(pred_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    prob_map: dict[tuple[str, str], float] = {}
    for _, row in pred_df.iterrows():
        p = row["p_blend_a"]
        if pd.isna(p):
            p = row["p_model_a"]
        if pd.isna(p):
            continue
        prob_map[(row["team_a_key"], row["team_b_key"])] = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return prob_map


def build_public_map(public_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    if public_df.empty:
        return out

    for _, row in public_df.iterrows():
        p = row["pct_pick_to_advance"]
        if pd.isna(p):
            p = row["pct_pick_to_reach_round"]
        if int(row["round_num"]) == 6 and pd.notna(row["pct_pick_to_win_title"]):
            p = row["pct_pick_to_win_title"]
        if pd.notna(p):
            out[(row["team_key"], int(row["round_num"]))] = float(np.clip(float(p), 1e-6, 1.0 - 1e-6))
    return out


def parse_payouts() -> dict[int, float]:
    if not PAYOUTS_ENV:
        return {1: 100.0}

    out: dict[int, float] = {}
    for token in PAYOUTS_ENV.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError("PAYOUTS must look like '1:100' or '1:70,2:20,3:10'")
        rank_s, amt_s = token.split(":", 1)
        out[int(rank_s.strip())] = float(amt_s.strip())

    if not out:
        raise ValueError("No valid payouts parsed from PAYOUTS.")
    return out


def p_win(prob_map: dict[tuple[str, str], float], team_a: str, team_b: str) -> float:
    direct = prob_map.get((team_a, team_b))
    if direct is not None:
        return float(np.clip(direct, 1e-6, 1.0 - 1e-6))

    reverse = prob_map.get((team_b, team_a))
    if reverse is not None:
        return float(np.clip(1.0 - reverse, 1e-6, 1.0 - 1e-6))

    return 0.5


def public_pick(public_map: dict[tuple[str, int], float], team_key: str, round_num: int) -> float:
    return float(public_map.get((team_key, round_num), np.nan))


def maybe_team_on_seed_line(
    rng: np.random.Generator,
    prob_map: dict[tuple[str, str], float],
    entrants: list[str],
) -> str:
    if len(entrants) == 1:
        return entrants[0]
    if len(entrants) == 2:
        p_a = p_win(prob_map, entrants[0], entrants[1])
        return entrants[0] if rng.random() < p_a else entrants[1]
    raise ValueError(f"Expected 1 or 2 entrants on seed line, found {entrants}")


def choose_true_winner(
    rng: np.random.Generator,
    prob_map: dict[tuple[str, str], float],
    team_a: str,
    team_b: str,
) -> str:
    return team_a if rng.random() < p_win(prob_map, team_a, team_b) else team_b


def build_true_bracket(
    rng: np.random.Generator,
    region_seed_map: dict[str, dict[int, list[str]]],
    region_order: list[str],
    semifinal_pairs: list[tuple[str, str]],
    prob_map: dict[tuple[str, str], float],
) -> dict[str, str]:
    picks: dict[str, str] = {}
    regional_champs: dict[str, str] = {}

    for region in region_order:
        seed_map = region_seed_map[region]
        entrants: dict[int, str] = {}

        for seed in range(1, 17):
            entrant = maybe_team_on_seed_line(rng, prob_map, seed_map[seed])
            entrants[seed] = entrant

        r64_winners: list[str] = []
        for idx, (high_seed, low_seed) in enumerate(OPENING_MATCHUPS, start=1):
            winner = choose_true_winner(rng, prob_map, entrants[high_seed], entrants[low_seed])
            picks[f"{region}_R64_{idx:02d}"] = winner
            r64_winners.append(winner)

        r32_winners: list[str] = []
        for idx, i in enumerate(range(0, 8, 2), start=1):
            winner = choose_true_winner(rng, prob_map, r64_winners[i], r64_winners[i + 1])
            picks[f"{region}_R32_{idx:02d}"] = winner
            r32_winners.append(winner)

        s16_winners: list[str] = []
        for idx, i in enumerate(range(0, 4, 2), start=1):
            winner = choose_true_winner(rng, prob_map, r32_winners[i], r32_winners[i + 1])
            picks[f"{region}_S16_{idx:02d}"] = winner
            s16_winners.append(winner)

        region_winner = choose_true_winner(rng, prob_map, s16_winners[0], s16_winners[1])
        picks[f"{region}_E8_01"] = region_winner
        regional_champs[region] = region_winner

    semifinal_winners: list[str] = []
    for idx, (region_a, region_b) in enumerate(semifinal_pairs, start=1):
        winner = choose_true_winner(rng, prob_map, regional_champs[region_a], regional_champs[region_b])
        picks[f"FINAL4_{idx:02d}"] = winner
        semifinal_winners.append(winner)

    champion = choose_true_winner(rng, prob_map, semifinal_winners[0], semifinal_winners[1])
    picks["TITLE"] = champion
    return picks


def round_num_from_slot(slot: str) -> int | None:
    if "_R64_" in slot:
        return 1
    if "_R32_" in slot:
        return 2
    if "_S16_" in slot:
        return 3
    if "_E8_" in slot:
        return 4
    if slot.startswith("FINAL4_"):
        return 5
    if slot == "TITLE":
        return 6
    return None


def score_bracket(candidate_picks: dict[str, str], true_picks: dict[str, str]) -> int:
    score = 0
    for slot, picked_team in candidate_picks.items():
        round_num = round_num_from_slot(slot)
        if round_num is None:
            continue
        if true_picks.get(slot) == picked_team:
            score += POINTS_BY_ROUND[round_num]
    return int(score)


def normalize_pair_choice(
    team_a: str,
    team_b: str,
    round_num: int,
    prob_map: dict[tuple[str, str], float],
    public_map: dict[tuple[str, int], float],
    style: str,
    rng: np.random.Generator,
) -> float:
    true_a = p_win(prob_map, team_a, team_b)
    public_a = public_pick(public_map, team_a, round_num)
    public_b = public_pick(public_map, team_b, round_num)

    if np.isnan(public_a) and np.isnan(public_b):
        public_ratio_a = true_a
    else:
        a = 0.0 if np.isnan(public_a) else public_a
        b = 0.0 if np.isnan(public_b) else public_b
        denom = a + b
        public_ratio_a = true_a if denom <= 0 else a / denom

    public_weight = OPP_PUBLIC_WEIGHT
    true_weight = OPP_TRUE_WEIGHT

    if style == "chalky":
        public_weight += 0.10
        true_weight -= 0.05
        public_ratio_a = np.clip(public_ratio_a * 1.05 + 0.5 * 0.0, 1e-6, 1 - 1e-6)
    elif style == "balanced":
        pass
    elif style == "contrarian":
        public_weight -= 0.20
        true_weight += 0.10
        public_ratio_a = 1.0 - public_ratio_a
    else:  # public
        pass

    mixed = public_weight * public_ratio_a + true_weight * true_a
    mixed += rng.normal(0.0, OPP_NOISE_STD)
    return float(np.clip(mixed, 1e-6, 1.0 - 1e-6))


def choose_opponent_winner(
    rng: np.random.Generator,
    team_a: str,
    team_b: str,
    round_num: int,
    prob_map: dict[tuple[str, str], float],
    public_map: dict[tuple[str, int], float],
    template: OpponentTemplate,
) -> str:
    q_a = normalize_pair_choice(
        team_a=team_a,
        team_b=team_b,
        round_num=round_num,
        prob_map=prob_map,
        public_map=public_map,
        style=template.style,
        rng=rng,
    )
    return team_a if rng.random() < q_a else team_b


def build_opponent_bracket(
    rng: np.random.Generator,
    region_seed_map: dict[str, dict[int, list[str]]],
    region_order: list[str],
    semifinal_pairs: list[tuple[str, str]],
    prob_map: dict[tuple[str, str], float],
    public_map: dict[tuple[str, int], float],
    template: OpponentTemplate,
) -> dict[str, str]:
    picks: dict[str, str] = {}
    regional_champs: dict[str, str] = {}

    for region in region_order:
        seed_map = region_seed_map[region]
        entrants: dict[int, str] = {}

        for seed in range(1, 17):
            entrants[seed] = maybe_team_on_seed_line(rng, prob_map, seed_map[seed])

        r64_winners: list[str] = []
        for idx, (high_seed, low_seed) in enumerate(OPENING_MATCHUPS, start=1):
            winner = choose_opponent_winner(
                rng=rng,
                team_a=entrants[high_seed],
                team_b=entrants[low_seed],
                round_num=1,
                prob_map=prob_map,
                public_map=public_map,
                template=template,
            )
            picks[f"{region}_R64_{idx:02d}"] = winner
            r64_winners.append(winner)

        r32_winners: list[str] = []
        for idx, i in enumerate(range(0, 8, 2), start=1):
            winner = choose_opponent_winner(
                rng=rng,
                team_a=r64_winners[i],
                team_b=r64_winners[i + 1],
                round_num=2,
                prob_map=prob_map,
                public_map=public_map,
                template=template,
            )
            picks[f"{region}_R32_{idx:02d}"] = winner
            r32_winners.append(winner)

        s16_winners: list[str] = []
        for idx, i in enumerate(range(0, 4, 2), start=1):
            winner = choose_opponent_winner(
                rng=rng,
                team_a=r32_winners[i],
                team_b=r32_winners[i + 1],
                round_num=3,
                prob_map=prob_map,
                public_map=public_map,
                template=template,
            )
            picks[f"{region}_S16_{idx:02d}"] = winner
            s16_winners.append(winner)

        region_winner = choose_opponent_winner(
            rng=rng,
            team_a=s16_winners[0],
            team_b=s16_winners[1],
            round_num=4,
            prob_map=prob_map,
            public_map=public_map,
            template=template,
        )
        picks[f"{region}_E8_01"] = region_winner
        regional_champs[region] = region_winner

    semifinal_winners: list[str] = []
    for idx, (region_a, region_b) in enumerate(semifinal_pairs, start=1):
        winner = choose_opponent_winner(
            rng=rng,
            team_a=regional_champs[region_a],
            team_b=regional_champs[region_b],
            round_num=5,
            prob_map=prob_map,
            public_map=public_map,
            template=template,
        )
        picks[f"FINAL4_{idx:02d}"] = winner
        semifinal_winners.append(winner)

    champion = choose_opponent_winner(
        rng=rng,
        team_a=semifinal_winners[0],
        team_b=semifinal_winners[1],
        round_num=6,
        prob_map=prob_map,
        public_map=public_map,
        template=template,
    )
    picks["TITLE"] = champion
    return picks


def choose_opponent_templates(pool_size: int, rng: np.random.Generator) -> list[OpponentTemplate]:
    styles = []
    for _ in range(pool_size - 1):
        x = rng.random()
        if x < 0.55:
            styles.append(OpponentTemplate(style="public"))
        elif x < 0.80:
            styles.append(OpponentTemplate(style="balanced"))
        elif x < 0.93:
            styles.append(OpponentTemplate(style="chalky"))
        else:
            styles.append(OpponentTemplate(style="contrarian"))
    return styles


def payout_share_for_rank(rank: int, payouts: dict[int, float]) -> float:
    return float(payouts.get(rank, 0.0))


def evaluate_single_sim(
    true_picks: dict[str, str],
    candidate_pick_maps: dict[str, dict[str, str]],
    opponent_pick_maps: list[dict[str, str]],
    payouts: dict[int, float],
) -> tuple[dict[str, int], dict[str, float], dict[str, bool], dict[str, bool]]:
    candidate_scores = {
        bracket_id: score_bracket(picks, true_picks)
        for bracket_id, picks in candidate_pick_maps.items()
    }
    opponent_scores = [score_bracket(picks, true_picks) for picks in opponent_pick_maps]

    point_accum = candidate_scores.copy()
    payout_accum: dict[str, float] = {}
    win_flags: dict[str, bool] = {}
    cash_flags: dict[str, bool] = {}
    top3_flags: dict[str, bool] = {}

    all_scores = opponent_scores.copy()

    for bracket_id, cand_score in candidate_scores.items():
        scores = [cand_score] + all_scores
        higher = sum(s > cand_score for s in scores)
        equal = sum(s == cand_score for s in scores)

        # Tied block occupies ranks [higher+1, higher+equal]
        payout = 0.0
        for r in range(higher + 1, higher + equal + 1):
            payout += payout_share_for_rank(r, payouts)
        payout /= max(equal, 1)

        payout_accum[bracket_id] = payout
        win_flags[bracket_id] = higher == 0
        cash_flags[bracket_id] = payout > 0
        top3_flags[bracket_id] = higher < 3

    return point_accum, payout_accum, win_flags, cash_flags, top3_flags


def persist_simulation_summary(
    con: duckdb.DuckDBPyConnection,
    summary_df: pd.DataFrame,
    sim_run_id: str,
) -> int:
    con.execute(
        "delete from marts.simulation_summary_fact where sim_run_id = ?",
        [sim_run_id],
    )

    payload = list(
        summary_df[
            [
                "sim_run_id",
                "bracket_id",
                "season",
                "n_sims",
                "expected_points",
                "variance_points",
                "first_place_prob",
                "cash_prob",
                "expected_payout",
            ]
        ].itertuples(index=False, name=None)
    )

    con.executemany(
        """
        insert into marts.simulation_summary_fact
        (sim_run_id, bracket_id, season, n_sims, expected_points, variance_points,
         first_place_prob, cash_prob, expected_payout)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )

    return len(payload)


def update_candidate_brackets(con: duckdb.DuckDBPyConnection, summary_df: pd.DataFrame) -> int:
    payload = list(
        summary_df[
            [
                "first_place_prob",
                "top_3_prob",
                "expected_payout",
                "bracket_id",
            ]
        ].itertuples(index=False, name=None)
    )

    con.executemany(
        """
        update marts.candidate_bracket
        set first_place_prob = ?,
            top_3_prob = ?,
            expected_payout = ?
        where bracket_id = ?
        """,
        payload,
    )
    return len(payload)


def main() -> int:
    con = connect_db()

    as_of_date = latest_prediction_date(con)
    pred_df = load_prediction_df(con, as_of_date)
    field_df = load_field(con)
    candidates_df = load_candidates(con)
    public_df = load_public_picks(con)

    payouts = parse_payouts()
    prob_map = build_prob_map(pred_df)
    public_map = build_public_map(public_df)

    region_order = parse_region_order(field_df)
    semifinal_pairs = parse_final_four_pairings(region_order)
    region_seed_map = build_region_seed_map(field_df)

    candidate_pick_maps = {
        row["bracket_id"]: json.loads(row["picks_json"])
        for _, row in candidates_df.iterrows()
    }

    rng = np.random.default_rng(EVAL_RANDOM_SEED)
    opponent_templates = choose_opponent_templates(POOL_SIZE, rng)

    score_history: dict[str, list[int]] = defaultdict(list)
    payout_history: dict[str, list[float]] = defaultdict(list)
    win_counts: dict[str, int] = defaultdict(int)
    cash_counts: dict[str, int] = defaultdict(int)
    top3_counts: dict[str, int] = defaultdict(int)

    for _ in range(EVAL_N):
        true_picks = build_true_bracket(
            rng=rng,
            region_seed_map=region_seed_map,
            region_order=region_order,
            semifinal_pairs=semifinal_pairs,
            prob_map=prob_map,
        )

        opponent_pick_maps = [
            build_opponent_bracket(
                rng=rng,
                region_seed_map=region_seed_map,
                region_order=region_order,
                semifinal_pairs=semifinal_pairs,
                prob_map=prob_map,
                public_map=public_map,
                template=template,
            )
            for template in opponent_templates
        ]

        point_accum, payout_accum, win_flags, cash_flags, top3_flags = evaluate_single_sim(
            true_picks=true_picks,
            candidate_pick_maps=candidate_pick_maps,
            opponent_pick_maps=opponent_pick_maps,
            payouts=payouts,
        )

        for bracket_id, points in point_accum.items():
            score_history[bracket_id].append(points)
            payout_history[bracket_id].append(payout_accum[bracket_id])
            win_counts[bracket_id] += int(win_flags[bracket_id])
            cash_counts[bracket_id] += int(cash_flags[bracket_id])
            top3_counts[bracket_id] += int(top3_flags[bracket_id])

    sim_run_id = f"{EVAL_MODEL_VERSION}_pool_{POOL_SIZE}_n{EVAL_N}_seed{EVAL_RANDOM_SEED}"

    rows: list[dict[str, Any]] = []
    for _, row in candidates_df.iterrows():
        bracket_id = row["bracket_id"]
        scores = np.array(score_history[bracket_id], dtype=float)
        payouts_arr = np.array(payout_history[bracket_id], dtype=float)

        rows.append(
            {
                "sim_run_id": sim_run_id,
                "bracket_id": bracket_id,
                "season": SEASON,
                "n_sims": EVAL_N,
                "expected_points": float(scores.mean()) if len(scores) else 0.0,
                "variance_points": float(scores.var(ddof=0)) if len(scores) else 0.0,
                "first_place_prob": win_counts[bracket_id] / EVAL_N,
                "cash_prob": cash_counts[bracket_id] / EVAL_N,
                "top_3_prob": top3_counts[bracket_id] / EVAL_N,
                "expected_payout": float(payouts_arr.mean()) if len(payouts_arr) else 0.0,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(
        ["first_place_prob", "expected_payout", "expected_points"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    n1 = persist_simulation_summary(con, summary_df, sim_run_id=sim_run_id)
    n2 = update_candidate_brackets(con, summary_df)

    print(f"Model version:   {EVAL_MODEL_VERSION}")
    print(f"As-of date:      {as_of_date}")
    print(f"Pool size:       {POOL_SIZE}")
    print(f"Simulations:     {EVAL_N}")
    print(f"Payouts:         {payouts}")
    print(f"Public picks:    {'yes' if public_map else 'no'}")
    print(f"sim_run_id:      {sim_run_id}")
    print(f"Summary rows:    {n1}")
    print(f"Bracket updates: {n2}")

    preview_cols = [
        "bracket_id",
        "first_place_prob",
        "top_3_prob",
        "cash_prob",
        "expected_payout",
        "expected_points",
        "variance_points",
    ]
    print("\nCandidate evaluation summary:")
    print(summary_df[preview_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())