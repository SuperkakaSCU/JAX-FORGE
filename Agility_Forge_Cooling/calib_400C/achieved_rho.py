"""Closure radius the press actually reached, per hit.

The action file records the COMMANDED rho.  The press cannot always deliver it:
its load saturates at 110.07 kN -- hits 5 and 7 both peak there to within
0.004 kN, and hit 7 sits within 1 % of that ceiling for 1.78 s against 0.28 s
for a hit that closes normally.

Machine position and commanded radius are otherwise related by a clean line.
Fitting it on the ten hits that never touch the ceiling gives a residual of
0.065 mm, so inverting that line recovers the radius each hit truly reached:
for those ten it returns the command, and for a stalled hit it returns where
the ram actually stopped.  Hit 7 stopped 1.43 mm short in machine position,
i.e. 0.60 mm short in radius -- simulating its commanded 16.36 mm asks for a
bite 0.6 mm deeper than the bar ever saw.
"""
import json
import numpy as onp
from paths import OUT, RAW, ACTIONS

CLIP_KN = float(__import__("os").environ.get("AGF_CLIP_KN", "110.0"))


def main():
    acts = []
    for line in ACTIONS.read_text().splitlines():
        if not line.strip():
            continue
        a = json.loads(line)["action"]
        if not a.get("no_op"):
            acts.append(a)
    acts = acts[:12]

    rho = onp.array([a["rho"] for a in acts])
    pos, peak = [], []
    for k in range(len(acts)):
        d = onp.load(OUT / f"press_hit_{k+1:02d}.npz")
        f = d["force_kn"]
        pos.append(float(d["position_mm"][int(onp.argmax(f))]))
        peak.append(float(f.max()))
    pos, peak = onp.array(pos), onp.array(peak)

    clean = peak < CLIP_KN
    A = onp.column_stack([rho[clean], onp.ones(clean.sum())])
    (m, c), *_ = onp.linalg.lstsq(A, pos[clean], rcond=None)
    resid = pos[clean] - (m * rho[clean] + c)
    rms = float(onp.sqrt(onp.mean(resid ** 2)))

    rho_eff = (pos - c) / m
    rows = []
    print(f"position = {m:.4f} * rho + {c:.3f}   (fit on {int(clean.sum())} unclipped "
          f"hits, residual RMS {rms:.4f} mm)\n")
    print(f"{'hit':>3} {'peak kN':>8} {'rho cmd':>8} {'rho eff':>8} {'short mm':>9} {'clipped':>8}")
    for k in range(len(acts)):
        short = float(rho_eff[k] - rho[k])
        rows.append(dict(hit=k + 1, peak_kn=peak[k], rho_command=float(rho[k]),
                         rho_achieved=float(rho_eff[k]), shortfall_mm=short,
                         clipped=bool(not clean[k])))
        print(f"{k+1:>3} {peak[k]:>8.2f} {rho[k]:>8.2f} {rho_eff[k]:>8.3f} "
              f"{short:>+9.3f} {'yes' if not clean[k] else '':>8}")

    out = dict(calibration=dict(slope=float(m), intercept=float(c),
                                residual_rms_mm=rms, clip_kn=CLIP_KN,
                                n_clean=int(clean.sum())),
               hits=rows)
    (OUT / "rho_achieved.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {OUT / 'rho_achieved.json'}")


if __name__ == "__main__":
    main()
