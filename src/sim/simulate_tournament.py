from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone
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

SIM_MODEL_VERSION = os.getenv(
    "SIM_MODEL_VERSION",
    os.getenv("OUTPUT_MODEL_VERSION", "logistic_seed_margin_v1_blend_v1"),
).strip()

SIM_N = int(os.getenv("SIM_N", "50000"))
SIM_SEED = int(os.getenv("SIM_SEED", "20260317"))

# Optional overrides.
# Example:
#   REGION_ORDER=East,West,South,Midwest
#   FINAL_FOUR_PAIRINGS=East-West,South-Midwest
REGION_ORDER_ENV = os.getenv("REGION_ORDER", "").strip()
FINAL_FOUR_PAIRINGS_ENV = os.getenv("FINAL_FOUR_PAIRINGS", "").strip()

OUTPUT_DIR = PROJECT_ROOT / "data" / "marts"

# Standard round-of-64 regional seed order
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

# Semantics for round_num written to marts.round_reach_prob_fact:
#   0 = won First Four / play-in
#   1 = won Round of 64
#   2 = won Round of 32
#   3 = won Sweet 16
#   4 = won Elite Eight (region champion / Final Four berth)
#   5 = won Final Four (title game berth)
#   6 = won national championship


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
        create table if not exists marts.round_reach_prob_fact (
            season integer not null,
            model_version varchar not null,
            as_of_date date not null,
            team_key varchar not null,
            round_num integer not null,
            reach_prob double not null,
            primary key (season, model_version, as_of_date, team_key, round_num)
        );
        """
    )

    return con


def latest_prediction_date(
    con: duckdb.DuckDBPyConnection,
    season: int,
    model_version: str,
) -> str:
    row = con.execute(
        """
        select cast(max(as_of_date) as varchar)
        from marts.matchup_prediction_fact
        where season = ?
          and model_version = ?
        """,
        [season, model_version],
    ).fetchone()

    if row is None or row[0] is None:
        raise RuntimeError(
            f"No matchup_prediction_fact rows found for season={season}, model_version={model_version}."
        )
    return str(row[0])


def load_probability_rows(
    con: duckdb.DuckDBPyConnection,
    season: int,
    model_version: str,
    as_of_date: str,
) -> pd.DataFrame:
    df = con.execute(
        """
        select
            team_a_key,
            team_b_key,
            p_model_a,
            p_market_a,
            p_blend_a
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


def build_prob_map(pred_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    prob_map: dict[tuple[str, str], float] = {}

    for _, row in pred_df.iterrows():
        p = row["p_blend_a"]
        if pd.isna(p):
            p = row["p_model_a"]

        if pd.isna(p):
            continue

        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        prob_map[(row["team_a_key"], row["team_b_key"])] = p

    return prob_map


def load_field(con: duckdb.DuckDBPyConnection, season: int) -> pd.DataFrame:
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
        order by region, seed, seed_label, team_key
        """,
        [season],
    ).df()

    if df.empty:
        raise RuntimeError("No staging.tournament_seed_dim rows found.")
    return df


def parse_region_order(field_df: pd.DataFrame) -> list[str]:
    regions = sorted(field_df["region"].dropna().astype(str).unique().tolist())

    if REGION_ORDER_ENV:
        requested = [x.strip() for x in REGION_ORDER_ENV.split(",") if x.strip()]
        missing = [r for r in requested if r not in regions]
        if missing:
            raise ValueError(f"REGION_ORDER contains unknown regions: {missing}. Known regions: {regions}")
        if set(requested) != set(regions):
            raise ValueError(f"REGION_ORDER must contain every region exactly once. Known regions: {regions}")
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
                raise ValueError(
                    "FINAL_FOUR_PAIRINGS must look like 'East-West,South-Midwest'."
                )
            a, b = [x.strip() for x in token.split("-", 1)]
            pairs.append((a, b))
            used.extend([a, b])

        if sorted(used) != sorted(region_order):
            raise ValueError(
                f"FINAL_FOUR_PAIRINGS must use each region exactly once. Regions: {region_order}"
            )
        return pairs

    if len(region_order) != 4:
        raise ValueError(
            "Automatic semifinal pairing assumes exactly 4 regions. "
            "Set FINAL_FOUR_PAIRINGS explicitly if your setup differs."
        )

    return [(region_order[0], region_order[1]), (region_order[2], region_order[3])]


def validate_field(field_df: pd.DataFrame, region_order: list[str]) -> None:
    n_teams = len(field_df)
    print(f"Tournament field rows: {n_teams}")
    print(f"Regions: {region_order}")

    summary = (
        field_df.groupby(["region", "seed"], as_index=False)
        .agg(n_teams=("team_key", "count"))
        .sort_values(["region", "seed"])
    )
    print("\nSeed counts by region:")
    print(summary.to_string(index=False))

    bad = summary[~summary["n_teams"].isin([1, 2])]
    if not bad.empty:
        raise ValueError(
            "Each region+seed should have either 1 team or 2 teams (for a play-in). "
            f"Bad rows:\n{bad}"
        )

    # Require 1..16 present in every region
    for region in region_order:
        present = set(
            field_df.loc[field_df["region"] == region, "seed"].astype(int).tolist()
        )
        missing = sorted(set(range(1, 17)) - present)
        if missing:
            raise ValueError(f"Region {region} is missing seed lines: {missing}")


def build_region_seed_map(field_df: pd.DataFrame) -> dict[str, dict[int, list[str]]]:
    out: dict[str, dict[int, list[str]]] = {}
    for region, g in field_df.groupby("region"):
        seed_map: dict[int, list[str]] = {}
        for seed, sg in g.groupby("seed"):
            seed_map[int(seed)] = sorted(sg["team_key"].astype(str).tolist())
        out[str(region)] = seed_map
    return out


def get_p_win(
    prob_map: dict[tuple[str, str], float],
    team_a: str,
    team_b: str,
    missing_pairs: set[tuple[str, str]],
) -> float:
    direct = prob_map.get((team_a, team_b))
    if direct is not None:
        return float(np.clip(direct, 1e-6, 1 - 1e-6))

    reverse = prob_map.get((team_b, team_a))
    if reverse is not None:
        return float(np.clip(1.0 - reverse, 1e-6, 1 - 1e-6))

    missing_pairs.add(tuple(sorted((team_a, team_b))))
    return 0.5


def play_game(
    rng: np.random.Generator,
    prob_map: dict[tuple[str, str], float],
    team_a: str,
    team_b: str,
    missing_pairs: set[tuple[str, str]],
) -> str:
    p_a = get_p_win(prob_map, team_a, team_b, missing_pairs)
    return team_a if rng.random() < p_a else team_b


def resolve_seed_entrant(
    rng: np.random.Generator,
    prob_map: dict[tuple[str, str], float],
    entrants: list[str],
    round_win_counts: dict[tuple[str, int], int],
    missing_pairs: set[tuple[str, str]],
) -> str:
    if len(entrants) == 1:
        return entrants[0]
    if len(entrants) == 2:
        winner = play_game(rng, prob_map, entrants[0], entrants[1], missing_pairs)
        round_win_counts[(winner, 0)] += 1
        return winner
    raise ValueError(f"Expected 1 or 2 entrants for a seed line, found {len(entrants)}: {entrants}")


def simulate_once(
    rng: np.random.Generator,
    region_seed_map: dict[str, dict[int, list[str]]],
    region_order: list[str],
    semifinal_pairs: list[tuple[str, str]],
    prob_map: dict[tuple[str, str], float],
    round_win_counts: dict[tuple[str, int], int],
    missing_pairs: set[tuple[str, str]],
) -> str:
    regional_champions: dict[str, str] = {}

    for region in region_order:
        seed_map = region_seed_map[region]

        # Round of 64
        r64_winners: list[str] = []
        for high_seed, low_seed in OPENING_MATCHUPS:
            team_a = resolve_seed_entrant(rng, prob_map, seed_map[high_seed], round_win_counts, missing_pairs)
            team_b = resolve_seed_entrant(rng, prob_map, seed_map[low_seed], round_win_counts, missing_pairs)
            winner = play_game(rng, prob_map, team_a, team_b, missing_pairs)
            round_win_counts[(winner, 1)] += 1
            r64_winners.append(winner)

        # Round of 32
        r32_winners: list[str] = []
        for i in range(0, 8, 2):
            winner = play_game(rng, prob_map, r64_winners[i], r64_winners[i + 1], missing_pairs)
            round_win_counts[(winner, 2)] += 1
            r32_winners.append(winner)

        # Sweet 16
        s16_winners: list[str] = []
        for i in range(0, 4, 2):
            winner = play_game(rng, prob_map, r32_winners[i], r32_winners[i + 1], missing_pairs)
            round_win_counts[(winner, 3)] += 1
            s16_winners.append(winner)

        # Elite Eight / region final
        region_winner = play_game(rng, prob_map, s16_winners[0], s16_winners[1], missing_pairs)
        round_win_counts[(region_winner, 4)] += 1
        regional_champions[region] = region_winner

    semifinal_winners: list[str] = []
    for region_a, region_b in semifinal_pairs:
        winner = play_game(
            rng,
            prob_map,
            regional_champions[region_a],
            regional_champions[region_b],
            missing_pairs,
        )
        round_win_counts[(winner, 5)] += 1
        semifinal_winners.append(winner)

    champion = play_game(rng, prob_map, semifinal_winners[0], semifinal_winners[1], missing_pairs)
    round_win_counts[(champion, 6)] += 1
    return champion


def build_round_prob_df(
    field_df: pd.DataFrame,
    round_win_counts: dict[tuple[str, int], int],
    n_sims: int,
    season: int,
    model_version: str,
    as_of_date: str,
) -> pd.DataFrame:
    teams = sorted(field_df["team_key"].astype(str).unique().tolist())
    rows: list[dict] = []

    for team_key in teams:
        for round_num in range(0, 7):
            count = round_win_counts.get((team_key, round_num), 0)
            rows.append(
                {
                    "season": season,
                    "model_version": model_version,
                    "as_of_date": pd.to_datetime(as_of_date).date(),
                    "team_key": team_key,
                    "round_num": round_num,
                    "reach_prob": count / n_sims,
                }
            )

    return pd.DataFrame(rows)


def persist_round_probs(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    season: int,
    model_version: str,
    as_of_date: str,
) -> int:
    con.execute(
        """
        delete from marts.round_reach_prob_fact
        where season = ?
          and model_version = ?
          and as_of_date = ?
        """,
        [season, model_version, as_of_date],
    )

    payload = list(
        df[
            ["season", "model_version", "as_of_date", "team_key", "round_num", "reach_prob"]
        ].itertuples(index=False, name=None)
    )

    con.executemany(
        """
        insert into marts.round_reach_prob_fact
        (season, model_version, as_of_date, team_key, round_num, reach_prob)
        values (?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def save_debug_outputs(
    round_prob_df: pd.DataFrame,
    champion_counts: dict[str, int],
    n_sims: int,
    model_version: str,
    as_of_date: str,
) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    round_path = OUTPUT_DIR / f"round_reach_probs_{model_version}_{ts}.csv"
    champ_path = OUTPUT_DIR / f"championship_probs_{model_version}_{ts}.csv"

    round_prob_df.to_csv(round_path, index=False)

    champ_df = pd.DataFrame(
        [
            {"team_key": team_key, "championship_prob": count / n_sims}
            for team_key, count in champion_counts.items()
        ]
    ).sort_values("championship_prob", ascending=False)

    champ_df.to_csv(champ_path, index=False)
    return round_path, champ_path


def print_summary(round_prob_df: pd.DataFrame) -> None:
    title_df = (
        round_prob_df[round_prob_df["round_num"] == 6]
        .sort_values("reach_prob", ascending=False)
        .head(20)
        .copy()
    )
    ff_df = (
        round_prob_df[round_prob_df["round_num"] == 4]
        .sort_values("reach_prob", ascending=False)
        .head(20)
        .copy()
    )

    print("\nTop championship probabilities:")
    print(title_df[["team_key", "reach_prob"]].to_string(index=False))

    print("\nTop Final Four probabilities:")
    print(ff_df[["team_key", "reach_prob"]].to_string(index=False))


def main() -> int:
    con = connect_db()

    as_of_date = latest_prediction_date(
        con=con,
        season=SEASON,
        model_version=SIM_MODEL_VERSION,
    )
    pred_df = load_probability_rows(
        con=con,
        season=SEASON,
        model_version=SIM_MODEL_VERSION,
        as_of_date=as_of_date,
    )
    prob_map = build_prob_map(pred_df)

    field_df = load_field(con=con, season=SEASON)
    region_order = parse_region_order(field_df)
    semifinal_pairs = parse_final_four_pairings(region_order)
    validate_field(field_df, region_order)
    region_seed_map = build_region_seed_map(field_df)

    print(f"\nUsing model_version: {SIM_MODEL_VERSION}")
    print(f"Prediction as_of_date: {as_of_date}")
    print(f"Simulations: {SIM_N}")
    print(f"Random seed: {SIM_SEED}")
    print(f"Semifinal pairings: {semifinal_pairs}")

    rng = np.random.default_rng(SIM_SEED)
    round_win_counts: dict[tuple[str, int], int] = defaultdict(int)
    champion_counts: dict[str, int] = defaultdict(int)
    missing_pairs: set[tuple[str, str]] = set()

    for _ in range(SIM_N):
        champion = simulate_once(
            rng=rng,
            region_seed_map=region_seed_map,
            region_order=region_order,
            semifinal_pairs=semifinal_pairs,
            prob_map=prob_map,
            round_win_counts=round_win_counts,
            missing_pairs=missing_pairs,
        )
        champion_counts[champion] += 1

    round_prob_df = build_round_prob_df(
        field_df=field_df,
        round_win_counts=round_win_counts,
        n_sims=SIM_N,
        season=SEASON,
        model_version=SIM_MODEL_VERSION,
        as_of_date=as_of_date,
    )
    n = persist_round_probs(
        con=con,
        df=round_prob_df,
        season=SEASON,
        model_version=SIM_MODEL_VERSION,
        as_of_date=as_of_date,
    )
    round_path, champ_path = save_debug_outputs(
        round_prob_df=round_prob_df,
        champion_counts=champion_counts,
        n_sims=SIM_N,
        model_version=SIM_MODEL_VERSION,
        as_of_date=as_of_date,
    )

    print(f"\nInserted marts.round_reach_prob_fact rows: {n}")
    print(f"Round probabilities CSV: {round_path}")
    print(f"Championship probabilities CSV: {champ_path}")

    if missing_pairs:
        sample_missing = sorted(missing_pairs)[:15]
        print("\nMatchups missing from matchup_prediction_fact. These defaulted to 0.5:")
        for pair in sample_missing:
            print(f"  - {pair[0]} vs {pair[1]}")
        if len(missing_pairs) > len(sample_missing):
            print(f"  ... and {len(missing_pairs) - len(sample_missing)} more")

    print_summary(round_prob_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())