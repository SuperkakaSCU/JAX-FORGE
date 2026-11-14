"""The two strokes each hit is actually made of, as closure radii.

The action file records one commanded rho per hit, but the press reaches it in
two goes: it closes part way, backs off a fraction of a millimetre, and closes
again onto the command.  The first pass always falls short, by an amount that
scales with the load -- 0.00457 mm per kN, r = 0.797 over the twelve hits, i.e.
about 4.6 um/kN of radial machine compliance.  The second pass lands on target
(mean error +0.0003 mm, no residual force dependence), which is what a
controller correcting for the first pass's error looks like.

That structure matters to the simulation only because the contact model is
fully sticking: a node that touches the die has all three displacement
components prescribed and cannot slide again for the rest of the hit.  Reality
releases that lock once per hit, at the retract; a single monotonic closure
never does.  This file supplies the radii needed to reproduce both passes.

Radii come from live_position_mm through the machine's own calibration line,
position = 2.3796 * rho - 7.390, fitted on the ten hits that never touch the
110.07 kN ceiling with a residual of 0.065 mm.
"""
import json

import numpy as onp

from paths import OUT, RAW, ACTIONS

F_THRESH = 5.0
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

    cal = json.loads((OUT / "rho_achieved.json").read_text())["calibration"]
    m, c = cal["slope"], cal["intercept"]
    rho_of = lambda p: (p - c) / m

    rows = []
    print(f"{'hit':>3} {'cmd rho':>8} | {'s1 peak':>8} {'s1 end':>8} {'short':>7} "
          f"| {'retract':>8} | {'s2 peak':>8} {'s2 end':>8} {'short':>7}")
    for k, a in enumerate(acts):
        d = onp.load(OUT / f"press_hit_{k+1:02d}.npz")
        f, p = d["force_kn"], d["position_mm"]
        idx = onp.where(f > F_THRESH)[0]
        segs = onp.split(idx, onp.where(onp.diff(idx) > 1)[0] + 1)
        if len(segs) < 2:
            print(f"{k+1:>3}  only {len(segs)} stroke(s) -- skipped")
            continue
        s1, s2 = segs[0], segs[1]
        r1, r2 = rho_of(p[s1].min()), rho_of(p[s2].min())
        # between the strokes the ram backs off; take the widest gap it opens
        between = slice(s1[-1], s2[0] + 1)
        rmid = rho_of(p[between].max()) if s2[0] > s1[-1] else r1
        rows.append(dict(hit=k + 1, rho_command=float(a["rho"]),
                         rho_stroke1=float(r1), rho_retract=float(rmid),
                         rho_stroke2=float(r2),
                         peak1_kn=float(f[s1].max()), peak2_kn=float(f[s2].max()),
                         clipped=bool(f.max() >= CLIP_KN)))
        r = rows[-1]
        print(f"{k+1:>3} {r['rho_command']:>8.2f} | {r['peak1_kn']:>8.2f} "
              f"{r['rho_stroke1']:>8.3f} {r['rho_stroke1']-r['rho_command']:>+7.3f} "
              f"| {r['rho_retract']:>8.3f} | {r['peak2_kn']:>8.2f} "
              f"{r['rho_stroke2']:>8.3f} {r['rho_stroke2']-r['rho_command']:>+7.3f}")

    (OUT / "stroke_rho.json").write_text(json.dumps(rows, indent=2))
    print(f"\n-> {OUT / 'stroke_rho.json'}")
    s1 = onp.array([r["rho_stroke1"] - r["rho_command"] for r in rows])
    print(f"stroke 1 falls short by {s1.mean():.3f} mm on average "
          f"({s1.min():.3f} to {s1.max():.3f})")


if __name__ == "__main__":
    main()
