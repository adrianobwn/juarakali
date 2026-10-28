"""Possession model: two frame classifiers + a 3-state Viterbi decoder.

    python solution/model.py dev      # leave-one-game-out on the Metrica sample games, tunes decoder
    python solution/model.py submit   # train on all sample games, label BAR-ATH, write the CSV

Trained and tuned only on other games (Metrica's public samples, labels derived from their human
event data). The provided match contributes no labels of any kind.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import features as ft  # noqa: E402
import possession_metrics as pm  # noqa: E402

GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")
PARAMS = os.path.join(HERE, "params.json")
BALL_STEMS = ("ball_", "bx", "d1_", "d2_", "n3_", "rel", "bvx", "dlog", "by_", "last_touch",
              "next_touch", "since_touch", "until_touch", "touch_bal")
SEED = 0
SEEDS = (0, 1, 2)  # degraded copies of each sample game: more ways for the ATD to lose the ball
DEAD_SEEN = 0.4  # share of stoppages where the ATD-like ball is visible at the restart spot


# -- data --------------------------------------------------------------------------------------

def degrade_ball(ball, label, seed, frac=0.12):
    """Blank the ball in blocks during live play, as the automatic tracking does.

    The ATD never flickers: it loses the ball in stretches. While players move fast (live play)
    it is missing ~10% of the time in gaps of median 6-9 s; most of its long gaps are stoppages,
    which the sample games already blank. This makes live play in the two look alike.
    """
    rng = np.random.RandomState(seed)
    ball = ball.copy()
    alive = np.isin(label, ["H", "A"])
    target = frac * alive.sum()
    masked = 0
    while masked < target:
        n = int(np.clip(rng.lognormal(np.log(7 * 25), 0.7), 25, 60 * 25))
        s = rng.randint(0, len(ball) - n)
        masked += (alive[s:s + n] & ~np.isnan(ball[s:s + n, 0])).sum()
        ball[s:s + n] = np.nan
    # The ATD often still sees a dead ball -- lying at the restart spot while players set up.
    # The samples blank it, so show it there for the tail of some stoppages.
    dead = (label == "D").astype(np.int8)
    edges = np.diff(np.concatenate(([0], dead, [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b >= len(ball) or np.isnan(ball[b, 0]) or rng.rand() > DEAD_SEEN:
            continue
        s = b - int((b - a) * rng.uniform(0.3, 1.0))
        ball[s:b] = ball[b] + rng.normal(0, 0.1, (b - s, 2))
    return ball + ball_noise(len(ball), rng)


def ball_noise(n, rng, s1=0.85, s2=3.5, p2=0.15):
    """ATD-like ball position error, metres: correlated jitter plus episodes of large error.

    The samples' corrected ball sits on the carrier's feet (0.07 m median when slow); the ATD
    puts it 1.06 m away (p25 0.57, p75 1.91, p90 4.5 m in live play). This reproduces that.
    """
    from scipy.signal import lfilter
    rho = np.exp(-1.0 / 12)  # about half a second of memory

    def ar1(sigma):
        return lfilter([np.sqrt(1 - rho ** 2)], [1, -rho], rng.normal(0, 1, (n, 2)), axis=0) * sigma
    big = np.zeros(n, dtype=bool)
    while big.mean() < p2:
        k = int(np.clip(rng.lognormal(np.log(25), 0.6), 5, 150))
        s = rng.randint(0, n - k)
        big[s:s + k] = True
    return ar1(s1) + np.where(big[:, None], ar1(s2), 0)


def sample_set(g, seed):
    import samples
    d = samples.load(g)
    halves = [tuple(np.flatnonzero(d["period"] == p)[[0, -1]] + [0, 1]) for p in (1, 2)]
    ball = degrade_ball(d["ball"], d["label"], seed)
    X, names = ft.frame_features(d["pl"], ball, d["team"], halves, d["pitch"])
    return dict(X=X, names=names, label=d["label"])


def cached(path, fn):
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    out = fn()
    np.savez(path, **out)
    return out


def load_samples():
    """{(game, seed): set}, one ATD-like degraded copy of each sample game per seed."""
    cache = os.path.join(ROOT, "data", "cache_samples")
    os.makedirs(cache, exist_ok=True)
    out = {}
    for g in GAMES:
        for s in SEEDS:
            d = cached(os.path.join(cache, "%s_s%d.npz" % (g, s)), lambda: sample_set(g, s))
            d["names"] = list(d["names"])
            d["seed"] = s
            out[g, s] = d
    return out


# -- classifiers -------------------------------------------------------------------------------

def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=SEED)


def ball_free(names):
    return np.array([not any(nm.startswith(s) for s in BALL_STEMS) for nm in names])


def train(sets, step=6):
    """(dead model, team model, names). Each frame also enters with the teams swapped."""
    names = sets[0]["names"]
    swap, flip = ft.swap_index(names)
    Xs, ys = [], []
    for d in sets:
        keep = np.flatnonzero(d["label"] != "X")[2 * int(d["seed"]) % step::step]
        Xs.append(d["X"][keep])
        ys.append(d["label"][keep])
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    Xsw = X[:, swap]
    Xsw[:, flip] *= -1
    ysw = np.where(y == "H", "A", np.where(y == "A", "H", y))
    XX = np.concatenate([X, Xsw])
    yy = np.concatenate([y, ysw])

    cols = ball_free(names)
    dead = gbm().fit(XX[:, cols], yy == "D")
    live = yy != "D"
    team = gbm().fit(XX[live], yy[live] == "H")
    return dead, team, names


def posteriors(models, X):
    dead, team, names = models
    pD = dead.predict_proba(X[:, ball_free(names)])[:, 1]
    pH = team.predict_proba(X)[:, 1]
    return pD, pH


# -- decoder -----------------------------------------------------------------------------------

def viterbi(E, cost):
    """argmax path over 3 states (H, A, D) for log-emissions E (n, 3); cost[i][j] for i -> j."""
    n = len(E)
    back = np.zeros((n, 3), dtype=np.int8)
    v0, v1, v2 = E[0]
    c = cost
    Et = E.tolist()
    for t in range(1, n):
        e0, e1, e2 = Et[t]
        # into H
        a, b, d = v0, v1 - c[1][0], v2 - c[2][0]
        if a >= b and a >= d:
            n0, b0 = a, 0
        elif b >= d:
            n0, b0 = b, 1
        else:
            n0, b0 = d, 2
        a, b, d = v0 - c[0][1], v1, v2 - c[2][1]
        if b >= a and b >= d:
            n1, b1 = b, 1
        elif a >= d:
            n1, b1 = a, 0
        else:
            n1, b1 = d, 2
        a, b, d = v0 - c[0][2], v1 - c[1][2], v2
        if d >= a and d >= b:
            n2, b2 = d, 2
        elif a >= b:
            n2, b2 = a, 0
        else:
            n2, b2 = b, 1
        back[t] = (b0, b1, b2)
        v0, v1, v2 = n0 + e0, n1 + e1, n2 + e2
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax([v0, v1, v2]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def presence_rates(pD, present, runs):
    """P(ball tracked | dead), P(ball tracked | live), estimated on this match without labels.

    The players-only dead model supplies the responsibilities, so the rates adapt to whatever
    tracker produced the ball -- the sample games blank it when dead, the ATD only mostly does.
    """
    m = np.zeros(len(pD), dtype=bool)
    for a, b in runs:
        m[a:b] = True
    r, q = pD[m], present[m]
    return (r * q).sum() / r.sum(), ((1 - r) * q).sum() / (1 - r).sum()


def emissions(pD, pH, ball_out, p, runs):
    eps = 1e-4
    pD = np.clip(pD, eps, 1 - eps)
    pH = np.clip(pH, eps, 1 - eps)
    E = np.column_stack([np.log1p(-pD) + np.log(pH), np.log1p(-pD) + np.log1p(-pH),
                         np.log(pD) + p["d_bias"]])
    out = np.nan_to_num(ball_out) > p["out_m"]
    E[out, 2] += p["out_bonus"]
    present = ~np.isnan(ball_out)
    qD, qL = np.clip(presence_rates(pD, present, runs), 0.02, 0.98)
    E[:, 2] += p["w_pres"] * np.where(present, np.log(qD), np.log1p(-qD))
    E[:, :2] += p["w_pres"] * np.where(present, np.log(qL), np.log1p(-qL))[:, None]
    return E


def decode(pD, pH, ball_out, runs, p):
    E = emissions(pD, pH, ball_out, p, runs)
    cost = [[0, p["c_team"], p["c_dead"]],
            [p["c_team"], 0, p["c_dead"]],
            [p["c_live"], p["c_live"], 0]]
    labels = np.full(len(pD), "X", dtype="<U1")
    for a, b in runs:
        labels[a:b] = np.array(["H", "A", "D"])[viterbi(E[a:b], cost)]
    return labels


def score(gt, pred):
    g, q = pm.label_codes(gt), pm.label_codes(pred)
    r = pm.evaluate(g, q)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


DEFAULT = dict(d_bias=0.5, out_m=1.0, out_bonus=3.0, c_team=8.0, c_dead=130.0, c_live=90.0,
               w_pres=0.0)


def tune(post, p0, rounds=3):
    """Coordinate search over the decoder parameters, maximising mean spellF1 over games."""
    # out_bonus is not tuned: the samples blank the ball as it leaves the pitch, so they cannot
    # measure it. It is fixed by the laws of the game -- a ball wholly over a line is out.
    grid = dict(d_bias=[-1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2],
                w_pres=[0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0],
                c_team=[2, 3, 5, 8, 12, 20, 30],
                c_dead=[3, 6, 10, 15, 20, 30, 45, 90],
                c_live=[3, 6, 10, 15, 20, 30, 45, 90])

    def obj(p):
        return np.mean([score(g, decode(pD, pH, bo, runs, p))[0]
                        for g, pD, pH, bo, runs in post])
    best, p = obj(p0), dict(p0)
    for _ in range(rounds):
        improved = False
        for k, vals in grid.items():
            for v in vals:
                if v == p[k]:
                    continue
                q = dict(p, **{k: v})
                s = obj(q)
                if s > best + 1e-4:
                    best, p, improved = s, q, True
            print("  %-9s -> %-6s  mean spellF1 %.4f" % (k, p[k], best), flush=True)
        if not improved:
            break
    return p, best


def logo_posteriors():
    """Leave-one-game-out posteriors for every sample game, cached in data/cache_dev.npy."""
    t0 = time.time()
    sets = load_samples()
    post = []
    for g in GAMES:
        models = train([d for (h, _), d in sets.items() if h != g])
        for s in SEEDS:
            d = sets[g, s]
            pD, pH = posteriors(models, d["X"])
            bo = d["X"][:, d["names"].index("ball_out_w0")]
            runs = pm.valid_runs(pm.label_codes(d["label"]))
            post.append((d["label"], pD, pH, bo, runs))
            lab = d["label"]
            live = np.isin(lab, ["H", "A"])
            print("%s seed %d  frame acc dead %.3f  team(live) %.3f  [%.0fs]" % (
                g, s, ((pD > .5) == (lab == "D"))[lab != "X"].mean(),
                ((pH > .5) == (lab == "H"))[live].mean(), time.time() - t0), flush=True)
    np.save(os.path.join(ROOT, "data", "cache_dev.npy"), np.array(post, dtype=object))
    return post


def dev(post=None, p0=DEFAULT):
    post = logo_posteriors() if post is None else post
    p, best = tune(post, p0)
    for k, (lab, pD, pH, bo, runs) in enumerate(post):
        print(GAMES[k // len(SEEDS)], "seed", SEEDS[k % len(SEEDS)],
              "spellF1 %.4f macroF1 %.4f  spells %d vs gt %d" % score(lab, decode(pD, pH, bo, runs, p)))
    with open(PARAMS, "w") as fh:
        json.dump(p, fh, indent=1)
    print("params ->", PARAMS, p)


def submit(name="juara_kali"):
    import atd
    from games import GAME, half_video_blocks
    p = json.load(open(PARAMS))
    models = train(list(load_samples().values()))
    pl, ball, team = atd.load()
    halves = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    X, names = ft.frame_features(pl, ball, team, halves, tuple(atd.PITCH))
    assert names == models[2]
    pD, pH = posteriors(models, X)
    ball_out = X[:, names.index("ball_out_w0")]
    print("ball tracked | dead %.2f, | live %.2f (estimated on this match, no labels)"
          % presence_rates(pD, ~np.isnan(ball_out), halves))
    labels = decode(pD, pH, ball_out, halves, p)
    # Before a half's opening kickoff the ball is not in play (Law 8). Only trusted where the
    # dead model agrees, so a kickoff after an early goal cannot wipe out real play.
    dirs = ft.attack_dirs(pl[..., :2].astype(np.float64), team, halves, atd.PITCH)
    for h, (a, b) in enumerate(halves):
        ko = ft.kickoff_frame(pl, team, (a, b), dirs[h], atd.PITCH)
        if ko > a and pD[a:ko].mean() > 0.5:
            labels[a:ko] = "D"
            print("half %d: dead before kickoff at frame %d" % (h + 1, ko))
    out = os.path.join(ROOT, "evaluation", "results", name + ".csv")
    with open(out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, l) for i, l in enumerate(labels))
    print("wrote", out, {k: int((labels == k).sum()) for k in "HADX"})


def retune():
    """Re-run only the decoder search on the cached leave-one-game-out posteriors."""
    post = list(np.load(os.path.join(ROOT, "data", "cache_dev.npy"), allow_pickle=True))
    dev(post, dict(json.load(open(PARAMS)), out_m=DEFAULT["out_m"], out_bonus=DEFAULT["out_bonus"]))


if __name__ == "__main__":
    {"dev": dev, "submit": submit, "retune": retune}[sys.argv[1]]()
