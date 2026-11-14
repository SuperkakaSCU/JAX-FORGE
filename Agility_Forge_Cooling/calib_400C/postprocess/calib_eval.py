"""Misfit of simulated vs measured force-closure curves on the calibration hits (read-only).

usage: calib_eval.py <calibration dir of the series> <sim dir> [<sim dir> ...] [--hits 1,2,3] [--json out.json]

Measured curves: press_hit_KK.npz mapped to the closure radius by geometry,
    rho(t) = rho_cmd + (p(t) - p_peak2) / 2 + C_R (F(t) - F_peak2),  C_R = 4.57 um/kN
(see force_curves.py), loading part of stroke 1 and stroke 2 (first 5 kN to the
force peak).  Simulated points: every committed step of stroke 1 and stroke 2
(history_travel_mm), closure = target rho + bite - travel.

Points used: measured force >= 30 kN at the simulated closure, inside the
measured closure range, and not a restart spike (a step whose force exceeds both
neighbours by more than 10 kN: the first step after an in-stroke remesh).  The
first ~0.2 mm of contact is left out on purpose: the simulation updates its
contact set at the end of a step, so the force lags one step at first contact.

Misfit per curve: RMS of (F_sim - F_meas) / P_meas, P_meas = that stroke's
measured peak force; the total is the mean over the curves (each hit and stroke
weighs the same).  Also printed: the factor lambda that best scales all
simulated forces at once (least squares on the same errors) and the misfit after
that scaling -- if sig0 and Q are both multiplied by lambda the forces scale by
about lambda (the elastic part does not), which the calibration search uses.
"""
import json
import sys
from pathlib import Path

import numpy as np

C_R, F_MIN, SPIKE = 0.00457, 30.0, 10.0
argv = sys.argv[1:]
HITS, JOUT = [1, 2, 3], None
if "--hits" in argv:
    i = argv.index("--hits"); HITS = [int(h) for h in argv[i + 1].split(",")]; del argv[i:i + 2]
if "--json" in argv:
    i = argv.index("--json"); JOUT = argv[i + 1]; del argv[i:i + 2]
CAL, SIMS = Path(argv[0]), argv[1:]
sr = {r["hit"]: r for r in json.loads((CAL / "stroke_rho.json").read_text())}


def measured(k):
    d = np.load(CAL / f"press_hit_{k:02d}.npz")
    f, p = d["force_kn"], d["position_mm"]
    idx = np.where(f > 5.0)[0]
    segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    j2 = segs[1][int(np.argmax(f[segs[1]]))]
    rho = sr[k]["rho_command"] + (p - p[j2]) / 2.0 + C_R * (f - f[j2])
    out = {}
    for st, s in zip(("stroke 1", "stroke 2"), segs[:2]):
        j = s[: int(np.argmax(f[s])) + 1]
        r, ff = rho[j], f[j]
        o = np.argsort(r)                                # interp needs increasing x
        out[st] = (r[o], ff[o])
    return out


def evaluate(sim):
    L = json.loads((Path(sim) / "sim_log_strokes.json").read_text())
    S = {(r["hit"], r["stroke"]): r for r in L if r["stroke"] in ("stroke 1", "stroke 2")}
    curves = []
    for k in HITS:
        M = measured(k)
        for st in ("stroke 1", "stroke 2"):
            r = S.get((k, st))
            if r is None or not r.get("history_travel_mm"):
                curves.append(dict(hit=k, stroke=st, missing=True))
                continue
            h = np.array(r["history_travel_mm"], dtype=float)
            rs, fs = r["rho"] + r["bite_mm"] - h[:, 0], h[:, 1]
            spike = np.zeros(len(fs), bool)
            for i in range(1, len(fs) - 1):
                spike[i] = fs[i] - max(fs[i - 1], fs[i + 1]) > SPIKE
            rm, fm = M[st]
            inside = (rs >= rm[0]) & (rs <= rm[-1])
            fmi = np.interp(rs, rm, fm)
            use = inside & ~spike & (fmi >= F_MIN)
            if use.sum() < 2:
                curves.append(dict(hit=k, stroke=st, missing=True))
                continue
            P = float(fm.max())
            a, b = fs[use] / P, fmi[use] / P                 # forces in units of the measured peak
            curves.append(dict(hit=k, stroke=st, n=int(use.sum()), n_spike=int(spike.sum()),
                               a=a.tolist(), b=b.tolist(), rms=float(np.sqrt(np.mean((a - b) ** 2))),
                               peak_sim=float(fs[-1]), peak_meas=float(fm.max())))
    ok = [c for c in curves if not c.get("missing")]
    total = float(np.mean([c["rms"] for c in ok])) if ok else float("nan")
    # best common scale: minimise sum_c mean_i (lam a_ci - b_ci)^2
    num = sum(np.mean(np.multiply(c["a"], c["b"])) for c in ok)
    den = sum(np.mean(np.square(c["a"])) for c in ok)
    lam = num / den if ok else float("nan")
    scaled = float(np.mean([np.sqrt(np.mean((lam * np.array(c["a"]) - np.array(c["b"])) ** 2)) for c in ok])) if ok else float("nan")
    return dict(sim=str(sim), curves=curves, misfit=total, scale=lam, misfit_scaled=scaled,
                complete=len(ok) == 2 * len(HITS))


res = [evaluate(s) for s in SIMS]
for r in res:
    print(f"== {r['sim']}")
    for c in r["curves"]:
        if c.get("missing"):
            print(f"   hit {c['hit']} {c['stroke']}: missing")
        else:
            print(f"   hit {c['hit']} {c['stroke']}: RMS {100 * c['rms']:5.1f} % of peak  (n {c['n']}, spikes {c['n_spike']}), "
                  f"peak sim {c['peak_sim']:5.1f} vs meas {c['peak_meas']:5.1f} kN")
    print(f"   misfit {100 * r['misfit']:.1f} %; best common scale {r['scale']:.3f} -> misfit {100 * r['misfit_scaled']:.1f} %"
          f"{'' if r['complete'] else '  (INCOMPLETE)'}")
if JOUT:
    Path(JOUT).write_text(json.dumps([{k: v for k, v in r.items() if k != "curves"} | {
        "curves": [{k: v for k, v in c.items() if k not in ("a", "b")} for c in r["curves"]]} for r in res], indent=1))
