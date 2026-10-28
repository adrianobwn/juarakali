"""Raw ATD -> numpy arrays, cached next to the data (data/ is gitignored)."""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAMES, GAME, require  # noqa: E402

PITCH = np.array(GAMES[GAME]["pitch"])  # metres, (length, width)


def load(source="tactical"):
    """players (n, slots, 3) and ball (n, 3) as x_m, y_m, speed; team (slots,) 0=H 1=A."""
    meta, raw = require(["atd_meta", "atd_raw"], source=source)
    cache = os.path.join(os.path.dirname(raw), "cache_" + os.path.basename(raw) + ".npz")
    root = ET.parse(meta).getroot()
    home = next(root.iter("Team")).get("id")
    team = np.array([p.get("teamId") != home for p in root.iter("Player")], dtype=np.int8)
    if not os.path.exists(cache):
        # "<frame>:<x,y,s>;...;<x,y,s>:<x,y,s>" per line -> one flat list of numbers
        with open(raw) as fh:
            lines = fh.read().split()
        vals = np.array(",".join(lines).replace(":", ",").replace(";", ",").split(","),
                        dtype=np.float64).reshape(len(lines), -1)
        assert (vals[:, 0] == np.arange(1, len(lines) + 1)).all(), "ATD frames not 1..n"
        np.savez(cache, vals=vals[:, 1:].astype(np.float32))
    vals = np.load(cache)["vals"]
    k = len(team)
    pl = vals[:, :3 * k].reshape(len(vals), k, 3).copy()
    ball = vals[:, 3 * k:3 * k + 3].copy()
    for a in (pl, ball):
        a[..., 0] *= PITCH[0]
        a[..., 1] *= PITCH[1]
    return pl, ball, team


if __name__ == "__main__":
    pl, ball, team = load()
    print(pl.shape, ball.shape, "home slots", int((team == 0).sum()))
