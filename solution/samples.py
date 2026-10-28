"""Metrica's public sample games (other matches, human event data) as a labelled dev set.

Labels are derived mechanically from the events, mirroring the challenge definition: a team owns
the ball from its last on-ball event; the ball is dead from the moment it goes out (BALL OUT /
SHOT ...OUT end frame), a goal, or a foul, until the next restart event.

    git clone https://github.com/metrica-sports/sample-data data/metrica-sample-data
    (commit e706dd5), or point MTAH_SAMPLES_DIR at its data/ folder.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.environ.get("MTAH_SAMPLES_DIR") or os.path.join(
    HERE, "..", "data", "metrica-sample-data", "data")
PITCH = np.array([105.0, 68.0])

ON_BALL = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST"}
RESTART = {"PASS", "SHOT", "SET PIECE", "BALL LOST"}


def _csv_tracking(g):
    out = []
    for side in ("Home", "Away"):
        df = pd.read_csv(os.path.join(SAMPLES, g, "{}_RawTrackingData_{}_Team.csv".format(g, side)),
                         skiprows=2)
        cols = [c for c in df.columns if c.startswith("Player")]
        xy = np.stack([df[cols].values, df[[df.columns[df.columns.get_loc(c) + 1] for c in cols]].values], -1)
        out.append((xy, df))
    (h, dfh), (a, _) = out
    ball = dfh[["Ball", dfh.columns[dfh.columns.get_loc("Ball") + 1]]].values
    pl = np.concatenate([h, a], 1)
    team = np.r_[np.zeros(h.shape[1]), np.ones(a.shape[1])].astype(np.int8)
    return pl, ball, team, dfh["Period"].values


def _csv_events(g):
    ev = pd.read_csv(os.path.join(SAMPLES, g, "{}_RawEventsData.csv".format(g)))
    return [(r.Team == "Away", r.Type, str(r.Subtype), int(r._5), int(r._7))
            for r in ev.itertuples()]


def _epts(g):
    from kloppy import metrica
    ds = metrica.load_tracking_epts(meta_data=os.path.join(SAMPLES, g, g + "_metadata.xml"),
                                    raw_data=os.path.join(SAMPLES, g, g + "_tracking.txt"),
                                    coordinates="metrica")
    df = ds.to_df()
    home = ds.metadata.teams[0]
    ids = [(p.player_id, p.team == home) for t in ds.metadata.teams for p in t.players
           if "{}_x".format(p.player_id) in df]
    pl = np.stack([df[["{}_x".format(i) for i, _ in ids]].values,
                   df[["{}_y".format(i) for i, _ in ids]].values], -1)
    team = np.array([0 if h else 1 for _, h in ids], dtype=np.int8)
    ball = df[["ball_x", "ball_y"]].values
    frames = df["frame_id"].values
    assert (np.diff(frames) == 1).all()
    period = df["period_id"].values
    d = json.load(open(os.path.join(SAMPLES, g, g + "_events.json")))["data"]

    def sub(e):
        s = e.get("subtypes")
        return "-".join(x["name"] for x in s) if isinstance(s, list) else (s or {}).get("name", "")
    events = [(e["team"]["name"] != home.name, e["type"]["name"], sub(e),
               e["start"]["frame"], e["end"]["frame"]) for e in d]
    return pl, ball, team, period, events, frames[0]


def labels_from_events(events, n, period, first_frame=1):
    """Per-frame H/A/D/X from on-ball events and dead-ball triggers. X outside periods.

    Once the ball is dead only a kick brings it back (a set piece, pass, shot or a lost ball):
    a goalkeeper collecting a ball that went out is a RECOVERY, not a restart.
    """
    marks = []  # (frame, order, code) with code 0=H 1=A 2=D
    dead, cur_period = True, None
    order = sorted(range(len(events)), key=lambda k: (events[k][3], k))
    for k in order:
        away, typ, subtype, start, end = events[k]
        start, end = start - first_frame, end - first_frame
        sub = subtype.upper()
        p = period[min(max(start, 0), n - 1)]
        if p != cur_period:
            dead, cur_period = True, p
        if dead:
            if typ in RESTART:
                dead = False
                marks.append((start, k, int(away)))
        elif typ in ON_BALL:
            marks.append((start, k, int(away)))
        if typ == "BALL OUT" or (typ == "SHOT" and ("OUT" in sub or "GOAL" in sub)):
            marks.append((max(end, start), k + 0.5, 2))
            dead = True
        if typ == "FAULT RECEIVED" or "OFFSIDE" in sub or (typ == "CHALLENGE" and "FAULT" in sub):
            marks.append((start, k + 0.5, 2))
            dead = True
    marks.sort()
    lab = np.full(n, 2, dtype=np.int8)  # dead before the first kickoff
    for (f, _, c), nxt in zip(marks, marks[1:] + [(n, 0, 0)]):
        lab[max(f, 0):max(nxt[0], 0)] = c
    out = np.array(["H", "A", "D"])[lab]
    out[period < 1] = "X"
    # Before each period's kickoff the ball is dead; after the last event of a period it is X.
    for p in np.unique(period[period >= 1]):
        idx = np.flatnonzero(period == p)
        starts = [m[0] for m in marks if idx[0] <= m[0] <= idx[-1]]
        if starts:
            out[idx[0]:min(starts)] = "D"
            out[max(starts):idx[-1] + 1] = "X"  # ponytail: tail after last event is unknown
    return out


def load(g):
    """dict: pl (n, slots, 2) metres, ball (n, 2) metres, team, period, label."""
    if g == "Sample_Game_3":
        pl, ball, team, period, events, f0 = _epts(g)
    else:
        pl, ball, team, period = _csv_tracking(g)
        events, f0 = _csv_events(g), 1
    pl = pl * PITCH
    ball = ball * PITCH
    period = np.nan_to_num(period).astype(int)
    lab = labels_from_events(events, len(ball), period, f0)
    return dict(pl=pl.astype(np.float32), ball=ball.astype(np.float32), team=team,
                period=period, label=lab, pitch=PITCH)


if __name__ == "__main__":
    for g in ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3"):
        d = load(g)
        lab = d["label"]
        sp = [(a, b) for a, b in zip(lab[:-1], lab[1:]) if a != b]
        n_poss = sum(1 for a, b in sp if b in "HA")
        print(g, d["pl"].shape, {k: round((lab == k).mean(), 3) for k in "HADX"},
              "possessions~", n_poss, "ball nan alive", round(np.isnan(d["ball"][np.isin(lab, ["H", "A"]), 0]).mean(), 3),
              "dead", round(np.isnan(d["ball"][lab == "D", 0]).mean(), 3))
