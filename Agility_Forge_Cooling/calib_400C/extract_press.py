"""Cut the press stream into per-hit windows for the 400 C series.

There is no manifest.json for these runs.  The mcap itself carries the
structure instead: hmr/forge/deform/u_taken is logged about one second BEFORE
the press closes for hit k, and hmr/scan/buffer about 25 s after it, when the
bar has been scanned.  So hit k occupies [u_taken_k - 2 s, min(next scan,
next u_taken) - 1 s).  Within that window the ram counts as loaded above 5 kN;
each loaded run is one stroke.  A normal hit has two (close part way, back
off, close onto the command), exactly as at room temperature.

Known anomaly: roll3.2 hit 2 has a single 79 kN stroke, a 5 kN blip, and no
scan afterwards -- the hit was interrupted.  The .pt file papers over it by
repeating hit 3's peak (102.17) in slot 2; the value here is the real one.
"""
import argparse, json
import numpy as np
from mcap.reader import make_reader
from paths import MCAP, OUT, SERIES

F_THRESH = 5.0


def read(path):
    t, f, s, p, ut, sc = [], [], [], [], [], []
    with open(path, "rb") as fh:
        for _, ch, msg in make_reader(fh).iter_messages(
                topics=["hmr/press/state", "hmr/forge/deform/u_taken", "hmr/scan/buffer"]):
            if ch.topic == "hmr/press/state":
                d = json.loads(msg.data)
                t.append(d["timestamp_ns"]); f.append(d["live_force_kn"])
                s.append(d["live_stroke_mm"]); p.append(d["live_position_mm"])
            elif ch.topic == "hmr/forge/deform/u_taken":
                ut.append((msg.log_time, json.loads(msg.data)))
            else:
                sc.append(msg.log_time)
    t = np.asarray(t, dtype=np.int64); o = np.argsort(t)
    return (t[o], np.asarray(f)[o], np.asarray(s)[o], np.asarray(p)[o],
            sorted(ut, key=lambda x: x[0]), np.sort(np.asarray(sc, dtype=np.int64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=float, default=0.5)
    args = ap.parse_args()
    print(f"[{SERIES}] reading {MCAP.name} ...")
    t, F, S, P, ut, sc = read(MCAP)
    print(f"  {len(t):,} press samples, {(t[-1]-t[0])/1e9/60:.1f} min, "
          f"{len(ut)} u_taken, {len(sc)} scans, force max {F.max():.2f} kN")
    rows, windows = [], []
    for k, (tu, act) in enumerate(ut):
        nxt = ([ut[k+1][0]] if k + 1 < len(ut) else []) + [x for x in sc if x > tu]
        lo = tu - int(2.0e9)
        hi = (min(nxt) if nxt else t[-1] + 1) - int(1.0e9)
        m = (t >= lo) & (t < hi)
        tt, ff, ss, pp = t[m], F[m], S[m], P[m]
        idx = np.where(ff > F_THRESH)[0]
        segs = [g for g in (np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(idx) else [])
                if len(g) > 3]
        if not segs:
            print(f"  hit {k+1:2d}: no load above {F_THRESH} kN -- skipped"); continue
        i0, i1 = segs[0][0], segs[-1][-1]
        kk = (tt >= tt[i0] - int(args.pad*1e9)) & (tt <= tt[i1] + int(args.pad*1e9))
        s_rel = ss[kk] - ss[kk][np.argmax(ff[kk] > F_THRESH)]
        np.savez(OUT / f"press_hit_{k+1:02d}.npz", hit=k+1, t_s=(tt[kk]-tt[kk][0])/1e9,
                 force_kn=ff[kk], stroke_mm=ss[kk], stroke_rel_mm=s_rel, position_mm=pp[kk])
        j = int(np.argmax(ff[kk]))
        rows.append(dict(hit=k+1, seq=act.get("seq"), phi=act.get("phi"), z=act.get("z"),
                         rho=act.get("rho"), n_samples=int(kk.sum()),
                         n_strokes=len(segs),
                         stroke_peaks_kn=[float(ff[g].max()) for g in segs],
                         stroke_durations_s=[float((tt[g[-1]]-tt[g[0]])/1e9) for g in segs],
                         peak_force_kn=float(ff[kk].max()),
                         stroke_at_peak_mm=float(ss[kk][j]),
                         position_at_peak_mm=float(pp[kk][j]),
                         work_kJ=float(np.trapezoid(ff[kk], s_rel)/1e3)))
        windows.append(dict(hit=k+1, start_ns=int(lo), end_ns=int(hi), u_taken_ns=int(tu)))
        r = rows[-1]
        flag = "" if r["n_strokes"] == 2 else f"   <-- {r['n_strokes']} stroke(s)"
        print(f"  hit {k+1:2d}: peaks {[round(x,1) for x in r['stroke_peaks_kn']]} kN  "
              f"dur {[round(x,2) for x in r['stroke_durations_s']]} s  "
              f"rho {r['rho']:.2f} phi {r['phi']:.1f} z {r['z']:.2f}{flag}")
    (OUT / "press_summary.json").write_text(json.dumps(rows, indent=2))
    (OUT / "hit_windows.json").write_text(json.dumps(windows, indent=2))
    print(f"wrote {len(rows)} hits -> {OUT}")


if __name__ == "__main__":
    main()
