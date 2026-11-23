# March Madness Bracket Optimizer

Which bracket gives you the best chance of winning *your pool*? This Python and DuckDB project estimates matchup probabilities, simulates the tournament, and evaluates candidate brackets against opponents modeled from public pick rates.

## Follow the code

| Stage | Implementation |
| --- | --- |
| Reconcile team names and provider IDs | [Team crosswalk](src/ingest/build_crosswalk.py) |
| Construct pregame matchup features | [Feature builder](src/features/build_matchup_features.py) |
| Train and evaluate the logistic model | [Training pipeline](src/models/train_logistic.py) |
| Simulate 50,000 tournaments by default | [Tournament simulation](src/sim/simulate_tournament.py) |
| Generate and score brackets | [Candidate generation](src/sim/generate_candidate_brackets.py) and [pool evaluation](src/sim/evaluate_pool.py) |

## Run with your own inputs

From the repository root, install the Python dependencies with `python -m pip install -r requirements.txt`. Create `data/` and `manual/`. The [DuckDB schema](sql/schema.sql) describes the tables. Run stages as modules, for example `python -m src.ingest.ingest_kaggle`. Obtain tournament history through your own Kaggle competition access. The CBD and market-odds adapters accept your own CSV or provider credentials as documented in each adapter.

The manual bracket and public-pick imports read `manual/2026_mens_bracket.csv` and `manual/public_picks_2026.csv`. Each loader declares its required columns. After loading inputs and resolving team names, build team-game and matchup features, train the logistic model, and select a consistent model version for blending and simulation. Source datasets, fitted models, API credentials and locally generated outputs are excluded.

Set `SIM_N` to vary the tournament sample size. Pool evaluation takes `POOL_SIZE`, round scoring, public-pick rates and a payout schedule into account. For example, `PAYOUTS=1:70,2:20,3:10` assigns payouts to the top three places. Compare expected payout alongside first-place probability and bracket duplication.
