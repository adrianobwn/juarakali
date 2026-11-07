# Juara Kali: ball possession, frame by frame

**More than a Hack 2027, Phase 1** · team *Juara Kali* · submission `juara_kali.csv` ·
code [github.com/adrianobwn/juarakali](https://github.com/adrianobwn/juarakali)

We label each frame with three gradient-boosted classifiers and a three-state Viterbi decoder.
The first classifier decides whether the ball is dead, judged only from how the players move.
The second decides which team has the ball. The third spots the moments when the ball changes
hands. All three are trained only on other matches: Metrica Sports' three public sample games,
with labels derived mechanically from their human event data. Before training, we degrade the
games' clean ball tracking with simulated automatic trackers drawn at random from wide ranges.
The decoder is tuned directly on `spellF1` with leave-one-game-out validation. BAR-ATH is only
the input: no label, tuned value or constant comes from it, and the public five minutes were never
used to train, tune or choose anything (§8).

| | spellF1 | macroF1 |
|---|---|---|
| leave-one-game-out, 3 other matches × 3 simulated trackers | **0.614** | 0.881 |
| official baseline (`baseline_smooth50`), same data | 0.293 | 0.674 |

---

## 1. Approach

`spellF1` scores possessions as whole objects, so the pipeline has to output clean segments
rather than per-frame guesses. We split the problem into frame probabilities and a decoder that
turns them into segments:

```
ATD ─► 244 features ─┬─► P(dead)          [players only] ─┐
                     ├─► P(home | live)   [all features] ─┤
                     └─► P(change | live) [all features] ─┼─► Viterbi (H/A/D) ─► CSV
       ball tracked in play? ball beyond a line? ─────────┘
```

Two rules from the Laws of the Game are applied on top: Law 9 (ball out of play) in the
decoder, and Law 8 (nothing is in play before the kickoff) after it.

### 1.1 Features (`solution/features.py`)

Each team's features are expressed in that team's own attacking frame, with x' running toward
the goal it attacks. Swapping the two team blocks and swapping H/A is then an exact
augmentation, and every training frame is used twice, once mirrored. As a result, nothing can
learn that "home has the ball more".

* **Players:** mean and median speed, the share of players below 1 m/s and above 4 m/s, each
  team's centroid, spread, deepest outfield line, most advanced player, forward velocity and
  share in the opponent half. Speeds are central differences over ±0.5 s, and jumps above
  12 m/s (identity switches) are discarded.
* **Ball, when tracked:** nearest and second-nearest distance per team, players within 3 m,
  ball speed, the ball's velocity relative to each team's nearest player (small when the ball
  is being carried), the ball's position in each team's frame and its distance outside the
  lines. We also include the last and next *touch* (ball within 1.0 m of a player; the radius was chosen from 1.0, 1.5
  and 2.0 m on the sample games, §5): which team,
  and how long ago or until.
* **How the ball moves:** a touch bends the ball's path. On a lightly smoothed track we measure
  the ball's acceleration, how sharply it turns and how much its pace changes. Each team also
  gets the acceleration measured while one of its players is within 2.5 m of the ball.
* Every feature is also averaged over centred windows of ±0.5, ±2, ±5 and ±10 s, and 2-second
  trends are added for speeds, forward velocities, centroids and the distance ratio.

### 1.2 Dead-ball model: players only

This is a `HistGradientBoostingClassifier` (600 trees, 63 leaves) on the 122 ball-free
features. The ball is left out on purpose. The training games blank the ball whenever it is
dead, while an automatic tracker need not, so ball features would teach the model that a
visible ball means live play. Player motion transfers cleanly between sources: in the samples,
median player speed is 1.9 m/s when the ball is live and 1.0 m/s when it is dead.

### 1.3 Team model and switch model

Both are trained on live frames with all 244 features. The **team model** outputs
P(home | live). The **switch model** outputs the probability that possession passes between the
teams within 0.25 s of the frame, that is, the moments where a decoder should be willing to
change its mind.

### 1.4 Decoder (`solution/model.py`)

The emissions are the log-probabilities of H, A and D. Two pieces of ball evidence are added:

* **Whether the ball is tracked in play.** P(tracked | dead) and P(tracked | live) are
  estimated separately for each match while it is being labelled, without labels, using the
  dead model's probabilities as responsibilities. The log-likelihood ratio enters with weight 3.
  A ball tracked beyond a line does not count as a ball in play.
* **Ball beyond a line.** A ball tracked more than 0.5 m outside the lines
  adds +8 per frame toward D (Law 9: a ball that has wholly crossed a line is out of play).

Transitions between H and A cost `max(−2 − 4·logit P(change), 0.5)` at each frame. A change of
possession is therefore nearly free where the switch model sees one, and expensive (about 10
nats when P(change) = 0.05) where it sees none. Going into D costs 200, coming out costs 10, and
D gets a +1 bias. Viterbi runs on each half separately. Every number in this section was
tuned on the sample games only (§5).

**Kickoff (Law 8).** The ball is not in play before a half's kickoff. The organisers publish both
kickoff frames (`starter/load_smart_tagging.py`, `FAQ.md`), and `FAQ.md` notes that the second
half is graded from frame 110,808 but kicks off at 113,742. Every frame of a half before its
published kickoff is set to D.

## 2. Training data

All training data comes from **other games**. We made no labels ourselves.

| source | what | labelled by | amount |
|---|---|---|---|
| [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) @ `e706dd5` | 3 anonymised professional matches: corrected tracking at 25 fps plus event data | Metrica Sports' human event annotation, published openly | 426,577 frames (4.7 h) |

Frame labels are derived mechanically from those events using the rules in §4. Three degraded
copies of each game are used (§2.1). Nothing about BAR-ATH was labelled by us, and the only
information about it we use beyond its tracking is what the organisers publish to every team:
the half boundaries and the two kickoff frames.

### 2.1 Simulated automatic trackers

The sample games come with corrected tracking: the ball sits on the carrier's feet (0.07 m median
when slow), is never lost in live play for long and is blanked whenever play stops. An automatic
tracker behaves differently, and a model trained on clean tracking would learn distance
thresholds of about 0.1 m that never fire on automatic data. We do not measure the provided
match's tracking. Each degraded copy instead gets its own simulated tracker, with every property
drawn from a wide range that runs from a nearly perfect tracker to a poor one:

| property of the simulated tracker | drawn from |
|---|---|
| live-play frames with no ball | 0–30%, in gaps of median 1–15 s (log-uniform), lognormal spread 1 |
| ball tracked on past the line after going out | median 0.2–3 s (log-uniform), along its last velocity |
| stoppages where the dead ball is seen at its restart spot | 0–100% |
| ball position error | AR(1) jitter σ 0.1–1.5 m with 0.2–1 s memory, plus spells of σ 1.5–5 m error on 0–30% of frames |

The ranges were set before any training and never tuned (tuning them on the same simulation would
be circular). As a check on difficulty, the official baseline scores spellF1 0.293 and macroF1
0.674 on our degraded samples, and 0.2433 and 0.6724 on BAR-ATH according to `EVALUATION.md`.

## 3. Assets used and dropped

* **Used:** the tactical ATD and its metadata (which slot belongs to which team), the published
  half boundaries in `games.py`, and the published kickoff frames.
* **Dropped: Smart Tagging.** No other game has Smart Tagging, so how much to trust it could not
  be set on external data.
* **Dropped: broadcast ATD and homography.** They are not synchronised with the tactical feed.
* **Dropped: video and tactical homography.** The ATD is already in pitch coordinates. The video
  was never used, neither as input nor to check our output.

## 4. Possession rules implemented

**Training labels, derived from the other games' events:**

* A team owns the ball from its last on-ball event: pass, carry, shot, recovery, set piece or
  lost ball. Duels (`CHALLENGE`) never change possession on their own.
* The ball dies when it crosses a line (at the end of `BALL OUT`, a shot that goes out, or a
  goal), at a foul (`FAULT RECEIVED`, or a `CHALLENGE` with `FAULT`) and at an offside.
* Once the ball is dead, only a kick restarts play: a set piece, pass, shot or lost ball. A
  goalkeeper picking up a ball that went out is a `RECOVERY`, not a restart. This rule fixed
  stoppages of 25–60 s that a naive derivation labels as live play.
* Before a period's kickoff, the ball is dead.

**Decoder:**

* Loose and contested balls stay with the last team. Possession flips only where the switch
  model sees a change, or where the other team's evidence persists, so a single touch,
  deflection or won header rarely flips it.
* Law 9: a ball tracked more than 0.5 m beyond a line pushes the decoder to D, and is
  never taken as a sign of live play.
* Law 8: nothing is in play before a half's published kickoff.
* At a restart, the team model decides which team resumes play.

## 5. Validation

The scorer is the repository's own `possession_metrics.py`. We ran leave-one-game-out over the
3 sample games with 3 degradations each, giving 9 evaluations. Decoder parameters were chosen by
block coordinate search on the mean `spellF1` of those 9. Parameters that trade off against each
other (the two switching weights, the two dead-ball costs, the three out-of-play settings) are
searched jointly.

| held-out game | spellF1 | macroF1 | possessions (pred / true) | baseline spellF1, same data |
|---|---|---|---|---|
| Sample 1 | 0.640 | 0.908 | 240 / 379 | 0.337 |
| Sample 2 | 0.620 | 0.858 | 247 / 354 | 0.282 |
| Sample 3 | 0.582 | 0.877 | 265 / 404 | 0.261 |
| **mean** | **0.614** | **0.881** | | **0.293** |

The touch radius was chosen the same way: 1.0 m scores 0.614, 1.5 m 0.611 and 2.0 m 0.608.
These scores are not comparable with those of our earlier versions, whose test copies were
degraded with a single simulated tracker.

**What each step was worth** (mean `spellF1` over the 9 evaluations, each step re-tuned; measured
with the earlier, single simulated tracker, and not re-run since):

| step | spellF1 |
|---|---|
| two classifiers + Viterbi | 0.606 |
| + ball-out fix (§1.4, §2.1) and ball-movement features | 0.611 |
| + switch model with per-frame switching costs | 0.625 |
| + larger trees, block-wise decoder search | **0.632** |

We also tried three additions and dropped them because none moved the mean by more than noise:
stop and restart detectors in the style of the switch model (+0.001), one-sided before/after
windows (+0.002) and stoppage-shape features such as a stationary player on the touchline
(−0.006).

**Where the error is** (earlier simulation). Replacing one model's output with the truth shows where the remaining
error lies. With a perfect team call, `spellF1` reaches 0.886; with a perfect dead-ball call,
0.688. Possessions of 8 s or more are recovered 81–86% of the time. Possessions under 2 s are
mostly missed (§6).

**Public five minutes.** We ran the public window through the official scorer as a pipeline
check only (frames aligned, H/A not swapped) and never used it to train, tune or choose. What we
learned from looking at it is listed in §8.

## 6. Limitations

* **Short possessions:** spells under about 2 s (duels, one-touch interceptions) are mostly
  missed, and we predict fewer possessions than there are. On those frames the team model is
  close to 50/50: with an automatic tracker's ball often a metre off the feet, a single touch
  rarely shows.
* **Quick restarts:** stoppages under 5 s are mostly not detected.
* **Training data:** only 3 matches from another competition, using corrected tracking that we
  degrade synthetically. A real tracker's failure modes are only approximated: a wrong object
  tracked as the ball, a track swapping teams, spare balls near the touchline (which could
  trigger the out-of-play rule).
* **With more time we would** train a sequence model (BiGRU) on the same features and add more
  external matches with automatic tracking.

## 7. Reproduction

```bash
git clone https://github.com/adrianobwn/juarakali && cd juarakali
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r solution/requirements.txt

# 1. gated match data: unpack into data/ as in DATA.md
#    (only data/tactical/BAR-ATH_tactical_atd.* is read)
# 2. public training games
git clone https://github.com/metrica-sports/sample-data data/metrica-sample-data
git -C data/metrica-sample-data checkout e706dd5

python solution/model.py submit    # ~3 min -> evaluation/results/juara_kali.csv
python evaluation/validate_submission.py evaluation/results/juara_kali.csv
```

Our file's md5 is `4d119bf9810c0794f39ff6815091c834`. We verified that rebuilding from empty
caches gives the identical file. `python solution/model.py dev` re-runs the leave-one-game-out
validation and rewrites `solution/params.json` (about 20 min). `retune` repeats only the decoder
search, using the cached probabilities.

## 8. What the provided match is used for

* **Input only.** `submit` reads the tactical ATD and its metadata, plus the half boundaries and
  kickoff frames that the organisers publish. Every model, threshold and decoder setting was
  trained or tuned on the three sample games. No value was set by measuring BAR-ATH.
* **Public five minutes.** Run only through the official scorer, as a pipeline check. Looking at
  it taught us two things about what counts as possession, which the rules allow. First,
  possessions shorter than a second count: our label rule already counted them, and we kept it.
  Second, a ball tracked behind a goal line is dead. That exposed a bug, because such a ball still
  counted as evidence of live play. We fixed the logic (§1.4) and set its numbers on the sample
  games only.
* **Video.** Never used: not as input, not to check output, not to tune.
