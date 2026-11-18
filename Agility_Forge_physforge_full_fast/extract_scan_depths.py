"""Extract the per-hit plastic reduction from the before/after scans in
2026-06-29.pt and export it as .npz.

`u0` in the record is the hammer stroke, not the plastic reduction: the measured
ratio u0/permanent-reduction ranges 0.18..1.17, so no single factor calibrates
it.  This script instead measures, for every hit, the change in section
thickness at the hammer position directly from vertices_k / vertices_k1, and
uses that as the driving quantity.

Thickness is the 2..98 percentile span rather than max-min, to suppress scan
outliers.  Both measures agree on the large reduction at hit 6 (~12 mm), which
confirms it is real and not noise.

Usage (from the jax-fem repository root):
    python -m applications.Agility_Forge_physforge_full_fast.extract_scan_depths
"""
import os
import numpy as onp

from applications.Agility_Forge_physforge_full_fast.read_pt import load_pt

PT = "applications/Agility_Forge/data/2026-06-29.pt"
BAND_HALF = 4.0     # +-4 mm slice around the hammer position (8 mm footprint)
PCT = 2.0           # percentile trim


def section_thickness(verts, n, x_center, axis, half=BAND_HALF, pct=PCT):
    """Robust section thickness along the pressing direction, in the slice
    under the hammer."""
    s = verts[:n]
    s = s[onp.abs(s[:, 0] - x_center) < half]
    if len(s) < 50:
        return onp.nan
    lo, hi = onp.percentile(s[:, axis], [pct, 100.0 - pct])
    return float(hi - lo)


def main():
    here = os.path.dirname(__file__)
    d = load_pt(PT)
    u = d["u"]
    vk, nk = d["vertices_k"], d["n_vertices_k"]
    vk1, nk1 = d["vertices_k1"], d["n_vertices_k1"]

    n_hits = len(u)
    total, before, after = [], [], []
    for i in range(n_hits):
        c = float(u[i, 2])
        rot = float(u[i, 1])
        axis = 2 if abs(rot) > 0.1 else 1     # rot=90deg -> press z; rot=0 -> press y
        ta = section_thickness(vk[i], nk[i], c, axis)
        tb = section_thickness(vk1[i], nk1[i], c, axis)
        before.append(ta)
        after.append(tb)
        total.append(ta - tb)

    total = onp.array(total, dtype=onp.float64)
    if not onp.all(onp.isfinite(total)):
        raise RuntimeError(f"extraction failed, non-finite values: {total}")
    if onp.any(total <= 0):
        raise RuntimeError(f"extracted a non-positive reduction: {total}")

    out = os.path.join(here, "data", "physforge_scan_depths.npz")
    onp.savez(
        out,
        hit=onp.arange(1, n_hits + 1),
        depth_total_mm=total,           # total section reduction for this hit
        depth_per_side_mm=total / 2.0,  # per-platen travel used by the BC
        thickness_before_mm=onp.array(before),
        thickness_after_mm=onp.array(after),
        u0_mm=onp.array(u[:, 0], dtype=onp.float64),
        rotation_deg=onp.degrees(onp.array(u[:, 1], dtype=onp.float64)),
        axial_pos_mm=onp.array(u[:, 2], dtype=onp.float64),
        force=onp.array(d["F"], dtype=onp.float64),
    )

    print(f"wrote {out}")
    print(f"{'hit':>4} {'u0':>7} {'t_before':>9} {'t_after':>8} {'reduction':>10} "
          f"{'per side':>9} {'/u0':>6}")
    for i in range(n_hits):
        print(f"{i+1:4d} {u[i,0]:7.2f} {before[i]:9.2f} {after[i]:8.2f} "
              f"{total[i]:10.2f} {total[i]/2:9.2f} {total[i]/u[i,0]:6.2f}")
    print(f"\nmean per side {(total/2).mean():.2f} mm  "
          f"(mean u0/2 as used by v1 {(u[:,0]/2).mean():.2f} mm, "
          f"i.e. v1 was {(u[:,0]/2).mean()/(total/2).mean():.2f}x the measured value)")


if __name__ == "__main__":
    main()
