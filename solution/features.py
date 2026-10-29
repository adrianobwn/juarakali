"""Per-frame features, computed identically for the ATD and for the Metrica sample games.

Everything is expressed per team in that team's own attacking frame (x' grows toward the goal it
attacks), so the two team blocks are interchangeable: swapping them and swapping H/A is an exact
augmentation, and nothing can learn "home has the ball more".
"""
import warnings

import numpy as np

# Frames with nobody tracked (half-time, camera cuts) give all-NaN slices: NaN is the answer.
warnings.filterwarnings("ignore", r"(All-NaN slice|Mean of empty slice|Degrees of freedom)",
                        RuntimeWarning)

FPS = 25
TOUCH_M = 1.5  # the ATD ball sits ~1 m off the carrier's feet
SIGNED = ("dlog", "last_touch", "next_touch", "touch_bal")


def roll(a, w):
    """Centred NaN-aware rolling mean over +/- w frames, any trailing shape."""
    if w == 0:
        return a
    a = np.asarray(a, dtype=np.float64)
    ok = ~np.isnan(a)
    z = np.zeros((1,) + a.shape[1:])
    cs = np.concatenate([z, np.cumsum(np.where(ok, a, 0), 0)])
    cc = np.concatenate([z, np.cumsum(ok, 0)])
    i = np.arange(len(a))
    lo, hi = np.clip(i - w, 0, len(a)), np.clip(i + w + 1, 0, len(a))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cs[hi] - cs[lo]) / (cc[hi] - cc[lo])


def shift(a, k):
    """a[t + k], NaN past the ends."""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def velocity(p, w):
    """Central-difference velocity over +/- w frames, m/s. p (n, ..., 2)."""
    return (shift(p, w) - shift(p, -w)) * (FPS / (2.0 * w))


def attack_dirs(pl, team, halves, pitch):
    """{half_index: (dir_team0, dir_team1)}, +1 = attacks toward x = L.

    The team defending the left goal is the one whose deepest player sits nearer x = 0 -- its
    goalkeeper. Uses the per-frame minimum x, so it needs no slot or role metadata.
    """
    out = {}
    for h, (a, b) in enumerate(halves):
        mins = [np.nanmedian(np.nanmin(pl[a:b, team == k, 0], 1)) for k in (0, 1)]
        out[h] = (1, -1) if mins[0] < mins[1] else (-1, 1)
    return out


def kickoff_frame(pl, team, half, direction, pitch, search_s=300, min_s=3, settle_s=10):
    """Video frame of a half's opening kickoff, or the half's first frame if none is found.

    Law 8: at the kick-off every player is in their own half. The kickoff is the end of the
    first stretch of at least min_s seconds with at most one player per team (the kicker) over
    the halfway line, which is followed by settle_s seconds of play with the teams mixed.
    """
    a, b = half
    b = min(b, a + search_s * FPS)
    L = pitch[0]
    ok = (~np.isnan(pl[a:b, :, 0])).sum(1) >= 14
    for k in (0, 1):
        x = pl[a:b, team == k, 0]
        xo = x if direction[k] > 0 else L - x
        with np.errstate(invalid="ignore"):
            ok &= (xo > L / 2 + 1).sum(1) <= 1
    ok = roll(ok.astype(float), FPS // 2) > 0.5          # ride over a tracking blip
    edges = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if e - s >= min_s * FPS and ok[e:e + settle_s * FPS].mean() < 0.2:
            return a + e
    return a


def frame_features(pl, ball, team, halves, pitch):
    """(n, f) float32 feature matrix and names. halves: [(start, stop)] frame spans.

    Frames outside the spans get the direction of the nearest span; they are never scored.
    """
    L, W = pitch
    n = len(ball)
    xy = pl[..., :2].astype(np.float64)
    bxy = ball[:, :2].astype(np.float64)
    dirs = attack_dirs(xy, team, halves, pitch)
    d = np.zeros((n, 2))
    edges = [0] + [(halves[i][1] + halves[i + 1][0]) // 2 for i in range(len(halves) - 1)] + [n]
    for h in range(len(halves)):
        d[edges[h]:edges[h + 1]] = dirs[h]

    pv = velocity(xy, 12)                       # players, +/- 0.5 s
    ps = np.hypot(pv[..., 0], pv[..., 1])
    ps[ps > 12] = np.nan                        # identity switch, not a sprint
    bv2 = velocity(bxy, 2)
    bv12 = velocity(bxy, 12)
    bs2 = np.hypot(bv2[:, 0], bv2[:, 1])
    bs12 = np.hypot(bv12[:, 0], bv12[:, 1])

    dist = np.hypot(xy[..., 0] - bxy[:, None, 0], xy[..., 1] - bxy[:, None, 1])
    F = {}
    F["spd_all"] = np.nanmean(ps, 1)
    F["spd_med"] = np.nanmedian(ps, 1)
    F["slow_frac"] = np.nanmean(np.where(np.isnan(ps), np.nan, ps < 1.0), 1)
    F["fast_frac"] = np.nanmean(np.where(np.isnan(ps), np.nan, ps > 4.0), 1)
    F["n_tracked"] = (~np.isnan(xy[..., 0])).sum(1).astype(float)
    with np.errstate(invalid="ignore"):
        out_d = np.maximum.reduce([-bxy[:, 0], bxy[:, 0] - L, -bxy[:, 1], bxy[:, 1] - W,
                                   np.zeros(n)])
    out_d[np.isnan(bxy[:, 0])] = np.nan
    F["ball_out"] = out_d
    F["ball_spd2"] = bs2
    F["ball_spd12"] = bs12
    F["ball_edge"] = np.minimum.reduce([bxy[:, 0], L - bxy[:, 0], bxy[:, 1], W - bxy[:, 1]])
    # A touch bends the ball's path: acceleration, turn and change of pace, on a lightly
    # smoothed track so the detector's jitter does not read as touches.
    sv = velocity(roll(bxy, 2), 3)
    acc = (shift(sv, 3) - shift(sv, -3)) * (FPS / 6.0)
    before, after = shift(sv, -6), shift(sv, 6)
    sb, sa = np.hypot(before[:, 0], before[:, 1]), np.hypot(after[:, 0], after[:, 1])
    F["ball_acc"] = np.hypot(acc[:, 0], acc[:, 1])
    F["ball_turn"] = 1 - (before * after).sum(1) / (sb * sa + 1e-6)
    F["ball_dspd"] = sa - sb

    for k in (0, 1):
        m = team == k
        x = np.where(d[:, k:k + 1] > 0, xy[:, m, 0], L - xy[:, m, 0])
        y = np.where(d[:, k:k + 1] > 0, xy[:, m, 1], W - xy[:, m, 1])
        vx = pv[:, m, 0] * d[:, k:k + 1]
        F["n%d" % k] = (~np.isnan(x)).sum(1).astype(float)
        F["cx%d" % k] = np.nanmean(x, 1)
        F["sx%d" % k] = np.nanstd(x, 1)
        F["sy%d" % k] = np.nanstd(y, 1)
        F["xmax%d" % k] = np.nanmax(x, 1)
        F["xmin%d" % k] = np.nanmin(np.where(x < 0.12 * L, np.nan, x), 1)  # last outfield line
        F["vx%d" % k] = np.nanmean(vx, 1)
        F["spd%d" % k] = np.nanmean(ps[:, m], 1)
        F["half_opp%d" % k] = np.nanmean(np.where(np.isnan(x), np.nan, x > L / 2), 1)
        bx = np.where(d[:, k] > 0, bxy[:, 0], L - bxy[:, 0])
        F["bx%d" % k] = bx
        dk = np.sort(np.where(np.isnan(dist[:, m]), np.inf, dist[:, m]), 1)
        d1 = dk[:, 0].copy()
        d2 = dk[:, 1].copy()
        d1[np.isinf(d1)] = np.nan
        d2[np.isinf(d2)] = np.nan
        F["d1_%d" % k] = d1
        F["d2_%d" % k] = d2
        F["n3_%d" % k] = np.where(np.isnan(bxy[:, 0]), np.nan, (dk < 3).sum(1))
        F["kink%d" % k] = np.where(d1 < 2.5, F["ball_acc"], np.where(np.isnan(d1), np.nan, 0))
        # Does the ball move with team k's nearest player? Small = carried or controlled.
        j = np.argmin(np.where(np.isnan(dist[:, m]), np.inf, dist[:, m]), 1)
        nv = pv[:, m][np.arange(n), j]
        F["rel%d" % k] = np.hypot(bv12[:, 0] - nv[:, 0], bv12[:, 1] - nv[:, 1])
        F["bvx%d" % k] = bv12[:, 0] * d[:, k]   # ball moving toward team k's target
    with np.errstate(invalid="ignore", divide="ignore"):
        F["dlog"] = np.log((F["d1_0"] + 0.3) / (F["d1_1"] + 0.3))
    F["by"] = np.abs(bxy[:, 1] - W / 2)

    # Touches: the ball within TOUCH_M of a player, signed +1 team 0 / -1 team 1. The last and
    # the next touch are what an event-based possession label is made of.
    with np.errstate(invalid="ignore"):
        a0, a1 = np.nan_to_num(F["d1_0"], nan=1e9), np.nan_to_num(F["d1_1"], nan=1e9)
        tt = np.where(np.minimum(a0, a1) < TOUCH_M, np.where(a0 <= a1, 1.0, -1.0), np.nan)
    i = np.arange(n)
    last = np.maximum.accumulate(np.where(np.isnan(tt), -1, i))
    nxt = np.minimum.accumulate(np.where(np.isnan(tt), n, i)[::-1])[::-1]
    T = {"last_touch": np.where(last >= 0, tt[np.maximum(last, 0)], np.nan),
         "next_touch": np.where(nxt < n, tt[np.minimum(nxt, n - 1)], np.nan),
         "since_touch": np.where(last >= 0, np.minimum(i - last, 750) / FPS, np.nan),
         "until_touch": np.where(nxt < n, np.minimum(nxt - i, 750) / FPS, np.nan)}
    for w in (12, 50):
        T["touch_bal_w%d" % w] = roll(np.nan_to_num(tt), w) * (2 * w + 1)

    base = list(F)
    cols, names = [], []
    for name in base:
        for w in (0, 12, 50, 125, 250):
            cols.append(roll(F[name], w))
            names.append("%s_w%d" % (name, w))
    for name, v in T.items():
        cols.append(v)
        names.append(name)
    # Change over time: future minus past means, 2 s each side.
    for name in ("spd_all", "spd0", "spd1", "vx0", "vx1", "cx0", "cx1", "dlog"):
        cols.append(roll(shift(F[name], 50), 50) - roll(shift(F[name], -50), 50))
        names.append(name + "_trend")
    X = np.column_stack(cols).astype(np.float32)
    return X, names


def swap_index(names):
    """Column permutation that exchanges the two team blocks (team 0 <-> team 1)."""
    def other(nm):
        for a, b in (("0", "1"), ("1", "0")):
            for stem in ("cx", "sx", "sy", "xmax", "xmin", "vx", "spd", "half_opp", "bx",
                         "d1_", "d2_", "n3_", "kink", "rel", "bvx", "n"):
                pre = stem + a
                if nm.startswith(pre) and (nm[len(pre):].startswith("_")):
                    return stem + b + nm[len(pre):]
        return nm
    idx = [names.index(other(nm)) for nm in names]
    flip = [nm.startswith(SIGNED) for nm in names]  # signed by team: changes sign
    return np.array(idx), np.array(flip)
