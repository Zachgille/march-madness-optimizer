from __future__ import annotations

import hashlib
import json
import os
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

CANDIDATE_MODEL_VERSION = os.getenv(
    "CANDIDATE_MODEL_VERSION",
    os.getenv("SIM_MODEL_VERSION", os.getenv("OUTPUT_MODEL_VERSION", "logistic_seed_margin_v1_blend_v1")),
).strip()

CANDIDATE_COUNT = int(os.getenv("CANDIDATE_COUNT", "5"))
CANDIDATE_RANDOM_SEED = int(os.getenv("CANDIDATE_RANDOM_SEED", "20260317"))
CANDIDATE_PUBLIC_SOURCE = os.getenv("CANDIDATE_PUBLIC_SOURCE", "yahoo").strip().lower()

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
class CandidateTemplate:
    name: str
    objective_type: str
    style: str
    champion_target: str | None = None
    note: str | None = None


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

    return con


def latest_as_of_date(con: duckdb.DuckDBPyConnection, table_name: str, model_version: str) -> str:
    row = con.execute(
        f"""
        select cast(max(as_of_date) as varchar)
        from {table_name}
        where season = ?
          and model_version = ?
        """,
        [SEASON, model_version],
    ).fetchone()

    if row is None or row[0] is None:
        raise RuntimeError(
            f"No rows found in {table_name} for season={SEASON}, model_version={model_version}."
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
        [SEASON, CANDIDATE_MODEL_VERSION, as_of_date],
    ).df()

    if df.empty:
        raise RuntimeError("No matchup predictions found.")
    return df


def load_round_probs(con: duckdb.DuckDBPyConnection, as_of_date: str) -> pd.DataFrame:
    df = con.execute(
        """
        select team_key, round_num, reach_prob
        from marts.round_reach_prob_fact
        where season = ?
          and model_version = ?
          and as_of_date = ?
        """,
        [SEASON, CANDIDATE_MODEL_VERSION, as_of_date],
    ).df()

    if df.empty:
        raise RuntimeError("No round reach probabilities found. Run simulate_tournament.py first.")
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
        raise RuntimeError("No tournament field found in staging.tournament_seed_dim.")
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
            [SEASON, CANDIDATE_PUBLIC_SOURCE],
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


def build_round_prob_map(round_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    return {
        (row["team_key"], int(row["round_num"])): float(row["reach_prob"])
        for _, row in round_df.iterrows()
    }


def build_public_map(public_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}

    if public_df.empty:
        return out

    for _, row in public_df.iterrows():
        team_key = row["team_key"]
        round_num = int(row["round_num"])

        p = row["pct_pick_to_advance"]
        if pd.isna(p):
            p = row["pct_pick_to_reach_round"]
        if round_num == 6 and pd.notna(row["pct_pick_to_win_title"]):
            p = row["pct_pick_to_win_title"]

        if pd.notna(p):
            out[(team_key, round_num)] = float(np.clip(float(p), 1e-6, 1.0 - 1e-6))

    return out


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


def p_win(prob_map: dict[tuple[str, str], float], team_a: str, team_b: str) -> float:
    direct = prob_map.get((team_a, team_b))
    if direct is not None:
        return float(np.clip(direct, 1e-6, 1.0 - 1e-6))

    reverse = prob_map.get((team_b, team_a))
    if reverse is not None:
        return float(np.clip(1.0 - reverse, 1e-6, 1.0 - 1e-6))

    return 0.5


def round_prob(round_prob_map: dict[tuple[str, int], float], team_key: str, round_num: int) -> float:
    return float(round_prob_map.get((team_key, round_num), 0.0))


def public_pick(public_map: dict[tuple[str, int], float], team_key: str, round_num: int) -> float | None:
    return public_map.get((team_key, round_num))


def leverage_score(
    round_prob_map: dict[tuple[str, int], float],
    public_map: dict[tuple[str, int], float],
    team_key: str,
    round_num: int,
) -> float:
    p_true = max(round_prob(round_prob_map, team_key, round_num), 1e-6)
    p_public = max(public_pick(public_map, team_key, round_num) or 1e-6, 1e-6)
    return float(np.log(p_true / p_public))


def style_weights(style: str, round_num: int) -> tuple[float, float, float]:
    if style == "chalk":
        return (1.00, 0.15, 0.00)
    if style == "balanced":
        if round_num <= 2:
            return (0.90, 0.15, 0.05)
        if round_num <= 4:
            return (0.78, 0.20, 0.12)
        return (0.70, 0.20, 0.20)
    if style == "contrarian":
        if round_num <= 2:
            return (0.82, 0.15, 0.08)
        if round_num <= 4:
            return (0.65, 0.20, 0.25)
        return (0.55, 0.20, 0.35)
    return (0.85, 0.15, 0.05)


def team_choice_score(
    prob_map: dict[tuple[str, str], float],
    round_prob_map: dict[tuple[str, int], float],
    public_map: dict[tuple[str, int], float],
    team_key: str,
    opp_key: str,
    round_num: int,
    style: str,
) -> float:
    wp = p_win(prob_map, team_key, opp_key)
    rp = round_prob(round_prob_map, team_key, round_num)
    lev = leverage_score(round_prob_map, public_map, team_key, round_num) if public_map else 0.0

    w_pair, w_round, w_lev = style_weights(style, round_num)
    return (w_pair * wp) + (w_round * rp) + (w_lev * lev)


def choose_from_seed_line(
    seed_teams: list[str],
    prob_map: dict[tuple[str, str], float],
    round_prob_map: dict[tuple[str, int], float],
    public_map: dict[tuple[str, int], float],
    template: CandidateTemplate,
    rng: np.random.Generator,
) -> tuple[str, dict[str, str]]:
    picks: dict[str, str] = {}

    if len(seed_teams) == 1:
        return seed_teams[0], picks

    if len(seed_teams) != 2:
        raise ValueError(f"Expected 1 or 2 teams on a seed line, found {seed_teams}")

    a, b = seed_teams
    if template.champion_target in {a, b}:
        winner = template.champion_target
    else:
        sa = team_choice_score(prob_map, round_prob_map, public_map, a, b, 0, template.style)
        sb = team_choice_score(prob_map, round_prob_map, public_map, b, a, 0, template.style)
        noise = rng.normal(0, 0.002)
        winner = a if sa + noise >= sb else b

    picks[f"PLAYIN_{a}_VS_{b}"] = winner
    return winner, picks


def choose_game_winner(
    team_a: str,
    team_b: str,
    round_num: int,
    template: CandidateTemplate,
    prob_map: dict[tuple[str, str], float],
    round_prob_map: dict[tuple[str, int], float],
    public_map: dict[tuple[str, int], float],
    rng: np.random.Generator,
) -> str:
    if template.champion_target in {team_a, team_b}:
        return template.champion_target

    sa = team_choice_score(prob_map, round_prob_map, public_map, team_a, team_b, round_num, template.style)
    sb = team_choice_score(prob_map, round_prob_map, public_map, team_b, team_a, round_num, template.style)

    # Small noise helps avoid identical outputs on near-ties for non-chalk styles
    noise_scale = 0.000 if template.style == "chalk" else 0.003
    noise = rng.normal(0, noise_scale)

    return team_a if sa + noise >= sb else team_b


def build_template_list(
    round_prob_map: dict[tuple[str, int], float],
    public_map: dict[tuple[str, int], float],
    field_df: pd.DataFrame,
) -> list[CandidateTemplate]:
    teams = sorted(field_df["team_key"].astype(str).unique().tolist())

    title_rank = sorted(
        teams,
        key=lambda t: round_prob(round_prob_map, t, 6),
        reverse=True,
    )

    if public_map:
        leverage_rank = sorted(
            teams,
            key=lambda t: (
                leverage_score(round_prob_map, public_map, t, 6),
                round_prob(round_prob_map, t, 6),
            ),
            reverse=True,
        )
    else:
        leverage_rank = title_rank.copy()

    top_title = [t for t in title_rank if round_prob(round_prob_map, t, 6) > 0]
    top_leverage = [t for t in leverage_rank if round_prob(round_prob_map, t, 6) > 0]

    def nth(lst: list[str], idx: int) -> str | None:
        return lst[idx] if idx < len(lst) else None

    templates = [
        CandidateTemplate(
            name="chalk_1",
            objective_type="expected_points",
            style="chalk",
            champion_target=nth(top_title, 0),
            note="Highest title probability",
        ),
        CandidateTemplate(
            name="balanced_1",
            objective_type="expected_points",
            style="balanced",
            champion_target=nth(top_leverage, 0) if public_map else nth(top_title, 1),
            note="Balanced bracket with title leverage",
        ),
        CandidateTemplate(
            name="contrarian_1",
            objective_type="first_place_prob",
            style="contrarian",
            champion_target=nth(top_leverage, 1) if public_map else nth(top_title, 2),
            note="More leverage in later rounds",
        ),
        CandidateTemplate(
            name="chalk_2",
            objective_type="expected_points",
            style="chalk",
            champion_target=nth(top_title, 1),
            note="Second title favorite",
        ),
        CandidateTemplate(
            name="contrarian_2",
            objective_type="first_place_prob",
            style="contrarian",
            champion_target=nth(top_leverage, 2) if public_map else nth(top_title, 3),
            note="Alternative leverage title path",
        ),
        CandidateTemplate(
            name="balanced_2",
            objective_type="expected_payout",
            style="balanced",
            champion_target=nth(top_leverage, 3) if public_map else nth(top_title, 4),
            note="Secondary balanced build",
        ),
    ]

    out: list[CandidateTemplate] = []
    seen = set()
    for t in templates:
        key = (t.style, t.champion_target)
        if t.champion_target is None:
            continue
        if key in seen:
            continue
        out.append(t)
        seen.add(key)

    return out[: max(1, CANDIDATE_COUNT)]


def expected_points_from_picks(
    picks: dict[str, str],
    round_prob_map: dict[tuple[str, int], float],
) -> float:
    total = 0.0
    for slot, team_key in picks.items():
        if slot.startswith("PLAYIN_"):
            continue

        if "_R64_" in slot:
            round_num = 1
        elif "_R32_" in slot:
            round_num = 2
        elif "_S16_" in slot:
            round_num = 3
        elif "_E8_" in slot:
            round_num = 4
        elif slot.startswith("FINAL4_"):
            round_num = 5
        elif slot == "TITLE":
            round_num = 6
        else:
            continue

        total += POINTS_BY_ROUND[round_num] * round_prob(round_prob_map, team_key, round_num)

    return float(total)


def duplication_score_from_picks(
    picks: dict[str, str],
    public_map: dict[tuple[str, int], float],
) -> float | None:
    if not public_map:
        return None

    weights = []
    values = []

    for slot, team_key in picks.items():
        if slot.startswith("PLAYIN_"):
            continue

        if "_R64_" in slot:
            round_num = 1
        elif "_R32_" in slot:
            round_num = 2
        elif "_S16_" in slot:
            round_num = 3
        elif "_E8_" in slot:
            round_num = 4
        elif slot.startswith("FINAL4_"):
            round_num = 5
        elif slot == "TITLE":
            round_num = 6
        else:
            continue

        p = public_pick(public_map, team_key, round_num)
        if p is not None:
            weights.append(POINTS_BY_ROUND[round_num])
            values.append(p)

    if not weights:
        return None

    return float(np.average(values, weights=weights))


def build_bracket_candidate(
    template: CandidateTemplate,
    region_seed_map: dict[str, dict[int, list[str]]],
    region_order: list[str],
    semifinal_pairs: list[tuple[str, str]],
    prob_map: dict[tuple[str, str], float],
    round_prob_map: dict[tuple[str, int], float],
    public_map: dict[tuple[str, int], float],
    rng: np.random.Generator,
) -> dict[str, Any]:
    picks: dict[str, str] = {}
    regional_champs: dict[str, str] = {}

    for region in region_order:
        seed_map = region_seed_map[region]

        entrants: dict[int, str] = {}
        for seed in range(1, 17):
            winner, playin_picks = choose_from_seed_line(
                seed_teams=seed_map[seed],
                prob_map=prob_map,
                round_prob_map=round_prob_map,
                public_map=public_map,
                template=template,
                rng=rng,
            )
            picks.update(playin_picks)
            entrants[seed] = winner

        r64_winners: list[str] = []
        for idx, (high_seed, low_seed) in enumerate(OPENING_MATCHUPS, start=1):
            team_a = entrants[high_seed]
            team_b = entrants[low_seed]
            winner = choose_game_winner(
                team_a=team_a,
                team_b=team_b,
                round_num=1,
                template=template,
                prob_map=prob_map,
                round_prob_map=round_prob_map,
                public_map=public_map,
                rng=rng,
            )
            picks[f"{region}_R64_{idx:02d}"] = winner
            r64_winners.append(winner)

        r32_winners: list[str] = []
        for idx, i in enumerate(range(0, 8, 2), start=1):
            winner = choose_game_winner(
                team_a=r64_winners[i],
                team_b=r64_winners[i + 1],
                round_num=2,
                template=template,
                prob_map=prob_map,
                round_prob_map=round_prob_map,
                public_map=public_map,
                rng=rng,
            )
            picks[f"{region}_R32_{idx:02d}"] = winner
            r32_winners.append(winner)

        s16_winners: list[str] = []
        for idx, i in enumerate(range(0, 4, 2), start=1):
            winner = choose_game_winner(
                team_a=r32_winners[i],
                team_b=r32_winners[i + 1],
                round_num=3,
                template=template,
                prob_map=prob_map,
                round_prob_map=round_prob_map,
                public_map=public_map,
                rng=rng,
            )
            picks[f"{region}_S16_{idx:02d}"] = winner
            s16_winners.append(winner)

        region_winner = choose_game_winner(
            team_a=s16_winners[0],
            team_b=s16_winners[1],
            round_num=4,
            template=template,
            prob_map=prob_map,
            round_prob_map=round_prob_map,
            public_map=public_map,
            rng=rng,
        )
        picks[f"{region}_E8_01"] = region_winner
        regional_champs[region] = region_winner

    semifinal_winners: list[str] = []
    semifinalists: list[str] = []

    for idx, (region_a, region_b) in enumerate(semifinal_pairs, start=1):
        team_a = regional_champs[region_a]
        team_b = regional_champs[region_b]
        semifinalists.extend([team_a, team_b])

        winner = choose_game_winner(
            team_a=team_a,
            team_b=team_b,
            round_num=5,
            template=template,
            prob_map=prob_map,
            round_prob_map=round_prob_map,
            public_map=public_map,
            rng=rng,
        )
        picks[f"FINAL4_{idx:02d}"] = winner
        semifinal_winners.append(winner)

    champion = choose_game_winner(
        team_a=semifinal_winners[0],
        team_b=semifinal_winners[1],
        round_num=6,
        template=template,
        prob_map=prob_map,
        round_prob_map=round_prob_map,
        public_map=public_map,
        rng=rng,
    )
    picks["TITLE"] = champion

    expected_points = expected_points_from_picks(picks, round_prob_map)
    duplication_score = duplication_score_from_picks(picks, public_map)

    finalists = semifinal_winners.copy()
    picks_json = json.dumps(picks, sort_keys=True)

    bracket_id = hashlib.sha256(
        f"{SEASON}|{CANDIDATE_MODEL_VERSION}|{template.name}|{template.style}|{template.champion_target}|{picks_json}".encode("utf-8")
    ).hexdigest()

    return {
        "bracket_id": bracket_id,
        "season": SEASON,
        "objective_type": template.objective_type,
        "model_version": CANDIDATE_MODEL_VERSION,
        "picks_json": picks_json,
        "champion_team_key": champion,
        "finalist_1_team_key": finalists[0] if len(finalists) > 0 else None,
        "finalist_2_team_key": finalists[1] if len(finalists) > 1 else None,
        "expected_points": expected_points,
        "first_place_prob": None,
        "top_3_prob": None,
        "expected_payout": None,
        "duplication_score": duplication_score,
        "notes": template.note or template.style,
    }


def persist_candidates(con: duckdb.DuckDBPyConnection, candidates_df: pd.DataFrame) -> int:
    if candidates_df.empty:
        return 0

    payload = list(
        candidates_df[
            [
                "bracket_id",
                "season",
                "objective_type",
                "model_version",
                "picks_json",
                "champion_team_key",
                "finalist_1_team_key",
                "finalist_2_team_key",
                "expected_points",
                "first_place_prob",
                "top_3_prob",
                "expected_payout",
                "duplication_score",
                "notes",
            ]
        ].itertuples(index=False, name=None)
    )

    con.executemany(
        """
        insert or replace into marts.candidate_bracket
        (bracket_id, season, objective_type, model_version, created_at,
         picks_json, champion_team_key, finalist_1_team_key, finalist_2_team_key,
         expected_points, first_place_prob, top_3_prob, expected_payout,
         duplication_score, notes)
        values (?, ?, ?, ?, current_timestamp, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )

    return len(payload)


def summarize_candidates(candidates_df: pd.DataFrame) -> None:
    if candidates_df.empty:
        print("No candidate brackets generated.")
        return

    cols = [
        "objective_type",
        "champion_team_key",
        "finalist_1_team_key",
        "finalist_2_team_key",
        "expected_points",
        "duplication_score",
        "notes",
    ]
    print("\nGenerated candidate brackets:")
    print(candidates_df[cols].to_string(index=False))


def main() -> int:
    con = connect_db()
    as_of_date = latest_as_of_date(con, "marts.round_reach_prob_fact", CANDIDATE_MODEL_VERSION)

    pred_df = load_prediction_df(con, as_of_date)
    round_df = load_round_probs(con, as_of_date)
    field_df = load_field(con)
    public_df = load_public_picks(con)

    prob_map = build_prob_map(pred_df)
    round_prob_map = build_round_prob_map(round_df)
    public_map = build_public_map(public_df)

    region_order = parse_region_order(field_df)
    semifinal_pairs = parse_final_four_pairings(region_order)
    region_seed_map = build_region_seed_map(field_df)

    templates = build_template_list(round_prob_map, public_map, field_df)
    rng = np.random.default_rng(CANDIDATE_RANDOM_SEED)

    candidates: list[dict[str, Any]] = []
    seen_json: set[str] = set()

    for template in templates:
        candidate = build_bracket_candidate(
            template=template,
            region_seed_map=region_seed_map,
            region_order=region_order,
            semifinal_pairs=semifinal_pairs,
            prob_map=prob_map,
            round_prob_map=round_prob_map,
            public_map=public_map,
            rng=rng,
        )

        if candidate["picks_json"] in seen_json:
            continue

        candidates.append(candidate)
        seen_json.add(candidate["picks_json"])

    candidates_df = pd.DataFrame(candidates)

    if candidates_df.empty:
        print("No unique candidate brackets were generated.")
        return 1

    candidates_df = candidates_df.sort_values(
        ["expected_points", "duplication_score"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    n = persist_candidates(con, candidates_df)

    print(f"Model version: {CANDIDATE_MODEL_VERSION}")
    print(f"As-of date:    {as_of_date}")
    print(f"Public picks:  {'yes' if public_map else 'no'}")
    print(f"Inserted/updated candidate brackets: {n}")

    summarize_candidates(candidates_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())