-- DuckDB schema for the March Madness project

create schema if not exists raw;
create schema if not exists staging;
create schema if not exists marts;

-- =========================================================
-- RAW INGEST METADATA
-- =========================================================

create table if not exists raw.raw_source_snapshot (
    snapshot_id varchar primary key,
    source_name varchar not null,         -- kaggle, ncaa_bracket, cbd, odds_api, yahoo_picks
    fetched_at timestamp not null,
    season integer not null,
    file_path varchar not null,
    checksum varchar,
    notes varchar
);

-- =========================================================
-- CORE DIMENSIONS
-- =========================================================

create table if not exists staging.team_dim (
    team_key varchar primary key,         -- canonical internal key, e.g. 2026_duke
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

create table if not exists staging.team_name_crosswalk (
    alias_name varchar not null,
    normalized_alias varchar not null,
    source_name varchar not null,
    season integer,
    team_key varchar not null,
    primary key (alias_name, source_name, season)
);

create table if not exists staging.game_dim (
    game_key varchar primary key,
    season integer not null,
    source_game_id varchar,
    game_date timestamp,
    neutral_site boolean,
    venue varchar,
    city varchar,
    state varchar,
    is_postseason boolean,
    tournament_name varchar,
    round_name varchar,
    round_num integer,                    -- 0 First Four, 1 R64, 2 R32, ... 6 Title
    region varchar,
    slot varchar,                         -- e.g. W1, W2, R1G01 etc.
    team1_key varchar,
    team2_key varchar,
    team1_score integer,
    team2_score integer,
    winner_team_key varchar,
    loser_team_key varchar
);

create table if not exists staging.tournament_slot_dim (
    season integer not null,
    slot varchar primary key,             -- e.g. EAST_R64_01
    round_num integer not null,
    region varchar,
    parent_slot_1 varchar,                -- previous slot feeding into this game
    parent_slot_2 varchar,
    seed_source_1 varchar,                -- e.g. EAST_1, EAST_16A
    seed_source_2 varchar,
    child_slot varchar,                   -- next game this feeds into
    child_position integer                -- 1 or 2
);

create table if not exists staging.tournament_seed_dim (
    season integer not null,
    team_key varchar not null,
    region varchar not null,
    seed integer not null,
    seed_label varchar not null,          -- "1", "11a", "11b", etc.
    play_in_group varchar,                -- null or group id
    is_play_in boolean default false,
    initial_slot varchar,
    primary key (season, team_key)
);

-- =========================================================
-- TEAM / GAME FACTS
-- =========================================================

create table if not exists staging.team_game_stats_fact (
    game_key varchar not null,
    team_key varchar not null,
    opponent_team_key varchar not null,
    is_home boolean,
    is_away boolean,
    is_neutral boolean,
    result varchar,                       -- W/L
    points integer,
    opp_points integer,
    fgm integer,
    fga integer,
    fg3m integer,
    fg3a integer,
    ftm integer,
    fta integer,
    orb integer,
    drb integer,
    trb integer,
    ast integer,
    stl integer,
    blk integer,
    tov integer,
    pf integer,
    poss_est double,
    ortg double,
    drtg double,
    efg double,
    tov_rate double,
    orb_rate double,
    ftr double,
    opp_efg double,
    opp_tov_rate double,
    opp_orb_rate double,
    opp_ftr double,
    primary key (game_key, team_key)
);

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

create table if not exists staging.market_odds_fact (
    odds_row_id varchar primary key,
    season integer not null,
    game_key varchar,
    source_name varchar not null,         -- odds_api
    bookmaker varchar not null,
    market_type varchar not null,         -- h2h, spreads, totals, futures
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

create table if not exists staging.public_pick_fact (
    season integer not null,
    source_name varchar not null,         -- yahoo, espn
    snapshot_ts timestamp not null,
    round_num integer not null,
    team_key varchar not null,
    pct_pick_to_advance double,           -- 0 to 1
    pct_pick_to_reach_round double,       -- if available
    pct_pick_to_win_title double,         -- if available
    primary key (season, source_name, snapshot_ts, round_num, team_key)
);

-- =========================================================
-- MODEL FEATURES / PREDICTIONS
-- =========================================================

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

create table if not exists marts.round_reach_prob_fact (
    season integer not null,
    model_version varchar not null,
    as_of_date date not null,
    team_key varchar not null,
    round_num integer not null,
    reach_prob double not null,
    primary key (season, model_version, as_of_date, team_key, round_num)
);

-- =========================================================
-- BRACKET / SIMULATION OUTPUTS
-- =========================================================

create table if not exists marts.candidate_bracket (
    bracket_id varchar primary key,
    season integer not null,
    objective_type varchar not null,      -- expected_points, expected_payout, first_place_prob
    model_version varchar not null,
    created_at timestamp not null,
    picks_json varchar not null,          -- JSON string: slot -> team_key
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

-- =========================================================
-- HELPFUL VIEWS
-- =========================================================

create or replace view marts.current_team_snapshot as
select *
from (
    select
        *,
        row_number() over (
            partition by season, team_key
            order by as_of_date desc
        ) as rn
    from staging.team_season_snapshot_fact
)
where rn = 1;

create or replace view marts.latest_public_picks as
select *
from (
    select
        *,
        row_number() over (
            partition by season, source_name, round_num, team_key
            order by snapshot_ts desc
        ) as rn
    from staging.public_pick_fact
)
where rn = 1;

create or replace view marts.latest_market_odds as
select *
from (
    select
        *,
        row_number() over (
            partition by season, game_key, bookmaker, market_type
            order by snapshot_ts desc
        ) as rn
    from staging.market_odds_fact
)
where rn = 1;