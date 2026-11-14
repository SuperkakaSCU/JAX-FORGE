"""Evaluate the F-bar "centre" gradients at the true element centroid.

What is wrong in the package
----------------------------
jax_fem_checkpoint.basis_new.get_shape_vals_and_grads_center tabulates the
shape functions at the reference point (0.5, 0.5, 0.5) for every 3D element.
That is the centre of the reference HEXAHEDRON [0, 1]^3.  The reference
tetrahedron is the corner simplex, whose centroid is (0.25, 0.25, 0.25); the
point (0.5, 0.5, 0.5) lies outside it (x + y + z = 1.5 > 1).

On TET4 this is invisible: the shape gradients are constant, so they are the
same wherever they are evaluated.  On TET10 the gradients are linear, the
"centre" deformation gradient F0 becomes an extrapolation from outside the
element, and the F-bar factor (det F0 / det F)^(1/3) in lib/constitutive.py
scales every quadrature point by a volume change that belongs to no part of
the element.  Measured: Newton diverges to NaN on the first step of hit 1
even at a 0.08 % load fraction, with linear solves accurate to 1e-10.

The fix
-------
fe.py imports the function by name, so rebinding
jax_fem_checkpoint.fe.get_shape_vals_and_grads_center BEFORE the Problem is
built replaces it for every FiniteElement created afterwards.  The
replacement evaluates at the centroid of the element's own reference cell
(the mean of basix's reference vertices): (0.25, 0.25, 0.25) for a
tetrahedron, (0.5, 0.5, 0.5) for a hexahedron -- unchanged there.  The package
itself is not edited.
"""
import numpy as onp


def install_centroid_center():
    import basix
    from jax_fem_checkpoint import fe as _fe
    from jax_fem_checkpoint import basis_new as _bn

    if getattr(_fe, "_agf_centroid_installed", False):
        return

    def get_shape_vals_and_grads_center(ele_type, gauss_order=None):
        (element_family, basix_ele, _face, gauss_order_default,
         degree, re_order) = _bn.get_elements(ele_type)
        if gauss_order is None:
            gauss_order = gauss_order_default
        quad_points, _ = basix.make_quadrature(basix_ele, gauss_order)
        element = basix.create_element(element_family, basix_ele, degree)
        centroid = onp.asarray(basix.geometry(basix_ele)).mean(axis=0)
        center = onp.repeat(centroid[None, :], len(quad_points), axis=0)
        vg = element.tabulate(1, center)[:, :, re_order, :]
        shape_values_c = vg[0, :, :, 0]
        shape_grads_ref_c = onp.transpose(vg[1:, :, :, 0], axes=(1, 2, 0))
        return shape_values_c, shape_grads_ref_c

    _fe.get_shape_vals_and_grads_center = get_shape_vals_and_grads_center
    _fe._agf_centroid_installed = True
