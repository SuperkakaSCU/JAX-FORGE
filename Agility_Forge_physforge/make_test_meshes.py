"""Generate billet meshes of several cross-sections for validating the
topological surface test.

All bars share the same axial length and roughly the same cross-sectional area
as the recorded billet, so the same hit sequence is meaningful on each. The bar
axis is x in every case, as the package requires.

Usage (from the repository root):
    python -m applications.Agility_Forge_physforge.make_test_meshes
"""
import os
import numpy as onp
import gmsh
import meshio

LX = 59.2569          # axial length of the recorded billet
LY = 40.4768          # recorded cross-section
LZ = 39.6570
R_EQ = 0.5 * (LY + LZ) / 2.0     # radius giving a comparable section


def _finish(out, dim=3, order=1, hex_mesh=False):
    gmsh.model.occ.synchronize()
    if hex_mesh:
        for s in gmsh.model.getEntities(2):
            gmsh.model.mesh.setRecombine(*s)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)   # HXT-based
    gmsh.model.mesh.generate(dim)
    gmsh.write(out)
    gmsh.clear()


def build(out_dir, size=4.0):
    os.makedirs(out_dir, exist_ok=True)
    made = []
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)

    # ---- 1. rectangular bar (matches the recorded billet) ----
    gmsh.model.add("rect")
    gmsh.model.occ.addBox(0, -LY / 2, -LZ / 2, LX, LY, LZ)
    p = os.path.join(out_dir, "bar_rect_tet.msh")
    _finish(p); made.append(p)

    # ---- 2. cylinder ----
    gmsh.model.add("cyl")
    gmsh.model.occ.addCylinder(0, 0, 0, LX, 0, 0, R_EQ)
    p = os.path.join(out_dir, "bar_cylinder_tet.msh")
    _finish(p); made.append(p)

    # ---- 3. rounded rectangle (box with filleted long edges) ----
    gmsh.model.add("round")
    b = gmsh.model.occ.addBox(0, -LY / 2, -LZ / 2, LX, LY, LZ)
    gmsh.model.occ.synchronize()
    axial = [e[1] for e in gmsh.model.getEntities(1)
             if abs(gmsh.model.occ.getCenterOfMass(1, e[1])[0] - LX / 2) < 1e-6]
    try:
        gmsh.model.occ.fillet([b], axial, [6.0])
    except Exception:
        pass                      # fall back to the sharp box if fillet fails
    p = os.path.join(out_dir, "bar_rounded_tet.msh")
    _finish(p); made.append(p)

    # ---- 4. tapered bar (square section shrinking along x) ----
    gmsh.model.add("taper")
    w0, w1 = LY / 2, LY / 4
    pts = [gmsh.model.occ.addPoint(0, -w0, -w0), gmsh.model.occ.addPoint(0, w0, -w0),
           gmsh.model.occ.addPoint(0, w0, w0), gmsh.model.occ.addPoint(0, -w0, w0)]
    ls = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    lo = gmsh.model.occ.addCurveLoop(ls)
    f0 = gmsh.model.occ.addPlaneSurface([lo])
    pts2 = [gmsh.model.occ.addPoint(LX, -w1, -w1), gmsh.model.occ.addPoint(LX, w1, -w1),
            gmsh.model.occ.addPoint(LX, w1, w1), gmsh.model.occ.addPoint(LX, -w1, w1)]
    ls2 = [gmsh.model.occ.addLine(pts2[i], pts2[(i + 1) % 4]) for i in range(4)]
    lo2 = gmsh.model.occ.addCurveLoop(ls2)
    f1 = gmsh.model.occ.addPlaneSurface([lo2])
    gmsh.model.occ.addThruSections([lo, lo2], makeSolid=True, makeRuled=True)
    gmsh.model.occ.remove([(2, f0), (2, f1)], recursive=False)
    p = os.path.join(out_dir, "bar_taper_tet.msh")
    _finish(p); made.append(p)

    # ---- 5. rectangular bar, HEX8 (structured) ----
    gmsh.model.add("recthex")
    gmsh.model.occ.addBox(0, -LY / 2, -LZ / 2, LX, LY, LZ)
    gmsh.model.occ.synchronize()
    for c, n in zip(gmsh.model.getEntities(1), [None]):
        pass
    for e in gmsh.model.getEntities(1):
        com = gmsh.model.occ.getCenterOfMass(1, e[1])
        n = 16 if abs(com[0] - LX / 2) < 1e-6 else 9
        gmsh.model.mesh.setTransfiniteCurve(e[1], n)
    for s in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(s[1])
        gmsh.model.mesh.setRecombine(*s)
    for v in gmsh.model.getEntities(3):
        gmsh.model.mesh.setTransfiniteVolume(v[1])
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(3)
    p = os.path.join(out_dir, "bar_rect_hex.msh")
    gmsh.write(p); gmsh.clear(); made.append(p)

    gmsh.finalize()

    print()
    for p in made:
        m = meshio.read(p)
        kinds = {k: len(v) for k, v in m.cells_dict.items() if k in ("tetra", "hexahedron")}
        pts = onp.asarray(m.points)
        print(f"  {os.path.basename(p):<26} {len(pts):>6} nodes  {kinds}  "
              f"x=[{pts[:,0].min():.1f}, {pts[:,0].max():.1f}]")
    return made


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    build(os.path.join(here, "data", "test_meshes"))
