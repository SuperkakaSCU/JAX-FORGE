"""HEX8 node ordering between meshio/VTK and jax_fem_checkpoint.

Why this exists
---------------
jax_fem_checkpoint.basis_new uses a non-standard HEX8 node order
(re_order = [0, 4, 6, 2, 1, 5, 7, 3]; the standard [0, 1, 3, 2, 4, 5, 7, 6]
is commented out next to it).  Read in the meshio/VTK/gmsh order, every cell
comes out MIRRORED: on a distorted test block the summed JxW was -0.9704
against an exact volume of +0.9704, every weight negative.  Residual and
tangent then both change sign, so a displacement-driven Newton solve can
still converge -- but the reaction force, i.e. the press load, flips sign, and
nothing warns about it.

The package expects VTK order with x and z swapped.  Permuting each cell with
VTK_TO_PKG fixes it (measured: summed JxW +0.9704, all weights positive,
linear-field gradients exact to 5e-15).  The permutation is its own inverse,
so the same array maps package-ordered cells back to VTK for writing files.

The package itself is not edited: cells are permuted before they are handed
to jax_fem_checkpoint.generate_mesh.Mesh, and back before any VTU is written.
"""
import numpy as onp

VTK_TO_PKG = onp.array([0, 4, 7, 3, 1, 5, 6, 2])
PKG_TO_VTK = VTK_TO_PKG                       # an involution: p[p] == identity
assert onp.array_equal(VTK_TO_PKG[VTK_TO_PKG], onp.arange(8))

# the six quadrilateral faces of a VTK-ordered hexahedron
VTK_FACES = ([0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
             [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7])


def to_package(cells_vtk):
    return onp.asarray(cells_vtk)[:, VTK_TO_PKG]


def to_vtk(cells_pkg):
    return onp.asarray(cells_pkg)[:, PKG_TO_VTK]


def hex_volumes(points, cells_vtk):
    """Exact trilinear-hex volume by a 5-tet split (positive for valid VTK cells)."""
    P = onp.asarray(points); C = onp.asarray(cells_vtk)
    v = onp.zeros(len(C))
    for a, b, c, d in ((0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 4, 6, 7), (1, 3, 4, 6)):
        v += onp.einsum("ij,ij->i", onp.cross(P[C[:, b]] - P[C[:, a]], P[C[:, c]] - P[C[:, a]]),
                        P[C[:, d]] - P[C[:, a]]) / 6.0
    return v


def split_hex(cells_vtk):
    """Each VTK hexahedron as 6 tetrahedra (post-processing only)."""
    C = onp.asarray(cells_vtk)
    tets = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
    return onp.stack([C[:, list(t)] for t in tets], axis=1).reshape(-1, 4)


def outer_surface_nodes_hex(points, cells_vtk):
    """Lateral boundary nodes of a VTK-ordered hex mesh (end caps dropped)."""
    C = onp.asarray(cells_vtk)
    faces = onp.concatenate([C[:, f] for f in VTK_FACES], axis=0)
    key = onp.sort(faces, axis=1)
    _, idx, cnt = onp.unique(key, axis=0, return_index=True, return_counts=True)
    nodes = onp.unique(faces[idx[cnt == 1]])
    x = onp.asarray(points)[nodes, 0]
    keep = (x > x.min() + 1e-6) & (x < x.max() - 1e-6)
    return nodes[keep]
