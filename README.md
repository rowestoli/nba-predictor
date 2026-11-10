# NBA stat predictor

Pick a player and a matchup to see a projected stat line for their next game (points, rebounds, assists, threes), then set any line to get over/under odds. Projections come from gradient-boosted models backtested on a season of games they never saw.

**Live demo:** https://rowestoli-nba-predictor.netlify.app

## Results

Scored on 3,397 games from the 2024–25 season, after training on 2015–2023 and calibrating on 2023–24:

| Stat | Average miss (model) | Average miss (season average) | Better by | Over/under picks right | Always taking the under |
| --- | --- | --- | --- | --- | --- |
| Points | 4.67 | 4.85 | 3.6% | 59% | 54% |
| Rebounds | 1.90 | 1.97 | 3.2% | 60% | 56% |
| Assists | 1.34 | 1.40 | 3.8% | 62% | 57% |
| 3-pointers | 0.92 | 0.95 | 3.6% | 68% | 64% |

Single-game stats are noisy, so a few percent better than the season average is a real gain. The over/under odds are calibrated: when the model says 60%, the over hits close to 60% of the time (the app plots this).

## How it works

**Pipeline** (`pipeline/`, Python, pandas, scikit-learn):

1. `features.py` builds one row per player-game using only what was known before tip-off: rolling 5- and 10-game averages, an exponentially weighted average, season-to-date and last-season averages, minutes and shot volume, rest days and back-to-backs, home or away, and the opponent's defense so far that season (shrunk toward average early on, computed date by date so a game never sees its own result).
2. `train.py` trains a Poisson gradient-boosted model per stat, scores it against the season-average baseline on unseen games, and records how real outcomes spread around each projection. That spread turns a projection into over/under probabilities.
3. It then retrains on everything and precomputes next-game projections for every player against all 29 opponents, home and away, as small JSON files.

**Web app** (`src/`, TypeScript, Vite): loads one player's file at a time, computes over/under odds in the browser from the stored outcome spreads (unit-tested against the Python output), and draws the charts in plain SVG. No server.

## Run it

```bash
pip install -r pipeline/requirements.txt
npm run train   # rebuild features, backtest, export public/data (about 30 s)

npm install
npm run dev     # http://localhost:5173
npm test
```

## Data

Game logs and season averages for 439 players, 2014–15 through November 14, 2024, scraped from ESPN for the original project. Projections are for each player's next game after that date.

## Background

This started as our CSCI 0320 (Software Engineering) term project at Brown, built by a team of four with a Java server, an R cleaning script and a React front end. I rebuilt it from scratch on my own: a single Python pipeline with an honest backtest, calibrated odds, and a static site anyone can open.
