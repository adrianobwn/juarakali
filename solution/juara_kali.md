# Juara Kali: ball possession, frame by frame

**More than a Hack 2027, Phase 1** · team *Juara Kali* · submission `juara_kali.csv` ·
code [github.com/adrianobwn/juarakali](https://github.com/adrianobwn/juarakali)

We label each frame with two gradient-boosted classifiers and a three-state Viterbi decoder. One
classifier decides whether the ball is dead, judged only from how the players move. The other
decides which team has the ball. Both are trained only on other matches: Metrica Sports' three
public sample games, with labels derived mechanically from their human event data. Before
training, we degrade those games until their ball tracking looks like BAR-ATH's automatic
tracking. The decoder's switching costs are tuned directly on `spellF1` with leave-one-game-out
validation. Nothing about BAR-ATH was labelled by hand, and the public five minutes were never
used to train, tune or choose anything.

| | spellF1 | macroF1 |
|---|---|---|
| leave-one-game-out, 3 other matches × 3 degradations | **0.606** | 0.872 |
| official baseline (`baseline_smooth50`), same data | 0.287 | 0.689 |

---

## 1. Approach

`spellF1` scores possessions as whole objects, so the pipeline has to output clean segments
rather than per-frame guesses. We split the problem into frame probabilities and a decoder that
turns them into segments:

```
ATD ─► 219 features ─┬─► P(dead)          [players only] ─┐
                     └─► P(home | live)   [all features] ─┼─► Viterbi (H/A/D) ─► CSV
       ball tracked? ball beyond a line? ─────────────────┘
```

Two rules from the Laws of the Game are applied on top: Law 9 (ball out of play) in the
decoder, and Law 8 (opening kickoff) after it.

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
  lines. We also include the last and next *touch* (ball within 1.5 m of a player): which team,
  and how long ago or until.
* Every feature is also averaged over centred windows of ±0.5, ±2, ±5 and ±10 s, and 2-second
  trends are added for speeds, forward velocities, centroids and the distance ratio.

### 1.2 Dead-ball model: players only

This is a `HistGradientBoostingClassifier` on the 122 ball-free features. The ball is left out
on purpose. The training games blank the ball whenever it is dead, while the ATD does not
always, so ball features would teach the model that a visible ball means live play. Player
motion transfers cleanly between the two sources. Median player speed in the ATD is 1.44 m/s;
in the samples it is 1.9 m/s when the ball is live and 1.0 m/s when it is dead.

### 1.3 Team model

This is a second `HistGradientBoostingClassifier`, trained on live frames with all 219
features. It outputs P(home | live).

### 1.4 Decoder (`solution/model.py`)

The emissions are the log-probabilities of H, A and D. Two pieces of ball evidence are added:

* **Whether the ball is tracked at all.** P(tracked | dead) and P(tracked | live) are estimated
  separately for each match, without labels, using the dead model's probabilities as
  responsibilities. On BAR-ATH they come out at 0.29 and 0.85. The log-likelihood ratio enters
  with weight 2.0.
* **Ball more than 1 m outside the lines.** This adds +3 per frame toward D (Law 9: a ball that
  has wholly crossed a line is out of play). The value is fixed, not tuned, because the sample
  games blank the ball as soon as it leaves the pitch and therefore cannot measure it.

The transition costs are 8 for H↔A, 6 into D and 30 out of D, with a +0.25 bias on D.
Viterbi runs on each half separately.

**Opening kickoff (Law 8).** At the kickoff every player is in their own half. We detect the
first ≥ 3 s stretch with at most one player per team over the halfway line, followed by mixed
play. Everything before it is set to D if the dead model agrees (mean P(dead) > 0.5), so the
rule can never remove real play. On BAR-ATH this fires only for the second half: the graded
block starts about two minutes before the kickoff, and the detector places the kickoff at
frame 113754.

## 2. Training data

All training data comes from **other games**. We made no labels ourselves.

| source | what | labelled by | amount |
|---|---|---|---|
| [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) @ `e706dd5` | 3 anonymised professional matches: corrected tracking at 25 fps plus event data | Metrica Sports' human event annotation, published openly | 426,577 frames (4.7 h) |

Frame labels are derived mechanically from those events using the rules in §4. Three degraded
copies of each game are used (§2.1). Nothing about BAR-ATH was labelled, by us or by anyone,
and no human-produced data about BAR-ATH was used.

### 2.1 Making the sample games look like the ATD

We measured each property on the BAR-ATH ATD without labels, using player speed as a proxy for
live play, and reproduced it in the samples:

| property | BAR-ATH ATD | samples as published | samples after degradation |
|---|---|---|---|
| ball missing during live play | ~10% of fast-play frames, gaps of median 6–9 s | ~3–10% | +12% of live frames, gaps of median 7 s |
| dead ball still tracked | 29% (estimated) | 1–15% | ball shown at the restart spot in 40% of stoppages → 28% |
| slow ball: distance to nearest player | median 1.06 m, p75 1.91 m | 0.07 m: the corrected ball sits on the carrier's feet | median 1.09 m, p75 1.64 m |

The ball error is modelled as AR(1) jitter (σ 0.85 m, about 0.5 s of memory) plus episodes of
larger error (σ 3.5 m, 15% of frames). The last row matters most. Without it, the team model
learns distance thresholds of about 0.1 m that never fire on the ATD. As a check on the
degradation, the official baseline scores 0.287 on the degraded samples and 0.2433 on BAR-ATH,
so the two are of similar difficulty.

## 3. Assets used and dropped

* **Used:** the tactical ATD and its metadata (which slot belongs to which team), and the
  published half boundaries in `games.py`.
* **Dropped: Smart Tagging.** Its team tag agrees with the team nearest the ball only 58–87% of
  the time, depending on the code. No other game has Smart Tagging, so its weight could not be
  calibrated without touching BAR-ATH labels.
* **Dropped: broadcast ATD and homography.** They are not synchronised with the tactical feed.
  During live play the tactical ATD already has the ball about 90% of the time, so aligning the
  broadcast was not worth it.
* **Dropped: video and tactical homography.** The ATD is already in pitch coordinates.

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

* Loose and contested balls stay with the last team. Possession flips only when the other team's
  evidence persists, so a single touch, deflection or won header does not flip it. A two-way
  excursion costs 2 × 8 nats.
* Law 9: a ball tracked more than 1 m beyond a line is dead.
* Law 8: nothing is in play before the opening kickoff.
* At a restart, the team model decides which team resumes play.

## 5. Validation

The scorer is the repository's own `possession_metrics.py`. We ran leave-one-game-out over the
3 sample games with 3 degradations each, giving 9 evaluations. Decoder parameters were chosen by
coordinate search on the mean `spellF1` of those 9.

| held-out game | spellF1 | macroF1 | possessions (pred / true) | baseline spellF1, same data |
|---|---|---|---|---|
| Sample 1 | 0.620 | 0.889 | 257 / 379 | 0.313 |
| Sample 2 | 0.616 | 0.857 | 250 / 354 | 0.285 |
| Sample 3 | 0.581 | 0.870 | 293 / 404 | 0.263 |
| **mean** | **0.606** | **0.872** | | **0.287** |

**Where the error is.** Replacing one model's output with the truth shows where the remaining
error lies. With a perfect team call, `spellF1` reaches about 0.87; with a perfect dead-ball
call, about 0.68.

**Public five minutes.** We used the public window as a pipeline check only: frames aligned,
H/A not swapped, labels meaning what we think. We ran it on the final file and never used it to
choose between versions. The result is spellF1 0.581, macroF1 0.878, 14 possessions predicted
against 17 true. With 17 possessions this window cannot rank anything, as `EVALUATION.md`
explains.

**On BAR-ATH** the file has 271 possessions of mean 12.2 s, a dead share of 48% of graded frames
and 61.6% live possession for Barcelona.

## 6. Limitations

* **Short possessions:** spells under about 2 s (duels, one-touch interceptions) are mostly
  missed, and we predict fewer possessions than there are.
* **Quick restarts:** stoppages under 5 s are mostly not detected.
* **Training data:** only 3 matches from another competition, using corrected tracking that we
  degrade synthetically. The ATD's real failure modes are only approximated: a wrong object
  tracked as the ball, a track swapping teams, spare balls near the touchline (which could
  trigger the out-of-play rule).
* **With more time we would** train a sequence model (BiGRU) on the same features,
  synchronise the broadcast ATD to fill ball gaps, and self-train on BAR-ATH with algorithmic
  pseudo-labels.

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

python solution/model.py submit    # ~2 min -> evaluation/results/juara_kali.csv
python evaluation/validate_submission.py evaluation/results/juara_kali.csv
```

Our file's md5 is `6b08e46de25854eabc51149046bb16d7`. We verified that rebuilding from empty
caches gives the identical file. `python solution/model.py dev` re-runs the leave-one-game-out
validation and rewrites `solution/params.json` (about half an hour). `retune` repeats only the
decoder search, using the cached probabilities.
