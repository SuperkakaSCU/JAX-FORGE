"""Force-based nodal contact between the bar and a rigid forging die.

Why this exists
---------------
The prescribed-displacement contact (lib/boundary_conditions.py) holds a
contact node ON the die plane with a Dirichlet condition: it cannot tell a
pushed node from a pulled one, it locks the circumferential component, and
the die is a flat band with a sharp edge and nothing beyond it.  On every
mesh of roll3.1 the run ended the same way, at the downstream band edge: the
last held surface node followed the plane while the material under it flowed
past the edge and bulged ABOVE the plane, where the model has no die, until
the surface element folded (mesh A hit 6, h20 hit 4, h14 hit 2 -- the finer
the mesh the earlier, because it resolves the step more sharply).  The real
die (data_400C/info/torm_tool_lib/Tool2.stl) has no such edge: a 7.08 mm flat,
a 3.175 mm fillet and an 83 deg flank.

Model
-----
Every lateral surface node of the bar is a contact candidate at every step.
In the die frame (x along the bar, y the press direction, z across the die)
the die face is the recess profile g(s) >= 0 of DieProfile at axial distance
s from the band centre: the lower die occupies y <= y_L - g(s), the upper
y >= y_R + g(s).  For a node at p the signed distance to the nearer die is

    d = (p_y - y_L + g(s)) / |(g', 1)|            lower die
    d = (y_R + g(s) - p_y) / |(g', 1)|            upper die

(exact for the flat, first order on the fillet).  The die pushes with a
penalty force along the face normal n and never pulls:

    f_n = k_n * max(0, -d) * n

so a node is released as soon as the gap opens, whatever the step did.  On
the tangent plane P = I - n n^T a Coulomb friction force acts, the same
elastic-stick / slip return map as lib/friction.py but in two directions:

    F_trial = P F_old - k_t * P (u - u_ref)        k_t = mu * N / delta
    F       = F_trial                               |F_trial| <= mu N  (stick)
            = mu N * F_trial / |F_trial|            otherwise           (slip)

with N the normal force of the LAST CONVERGED step (lagged, so the slip limit
is fixed during a Newton solve), u_ref and F_old the committed state.  Both
forces enter the residual as external nodal forces and their derivatives
(k_n n n^T on the normal, the consistent stick / slip tangent on the
friction) as 3x3 blocks appended to the package's sparse triplets, through
ThermalMechanical._nodal_force_terms in lib/constitutive.py.  The press
load is the sum of the normal forces on each die.

Units: mm, N, MPa.
"""
import numpy as onp


class DieProfile:
    """Axial recess g(s) of the die face from its flat plane, and dg/ds.

    kind = "flat"    : 0 on |s| <= a, no die beyond (the sharp band)
           "fillet"  : flat, tangent circular arc of radius r up to the flank
                       angle, then a straight flank up to `height` (Tool2:
                       a = 3.54, r = 3.175, flank 83 deg, height 25.4)
           "sigmoid" : logistic wall of width w outside the flat, rising
                       smoothly from the flat edge to `height` (w = 0.5 mm
                       reaches 2.5 mm at 3.2 mm past the edge, like Tool2)
    g = +inf where there is no die (above `height`).
    """

    def __init__(self, kind, half_flat, fillet_r=3.175, flank_deg=83.0,
                 height=25.4, sigmoid_w=1.0, soft_w=0.0):
        self.kind = kind
        self.a = float(half_flat)
        # Soft edge of the "flat" profile: instead of a die face that ends at
        # |s| = a, the face stays flat and the contact STIFFNESS fades out
        # through the logistic window W(s) = S((a - |s|) / soft_w), 0.5 at
        # the nominal edge.  Any die face that simply ends is ill-posed for a
        # penalty force -- a node crossing the end loses a finite force at
        # once, and Newton cannot converge at any step size (measured: the
        # truncated shoulder on mesh A stalled at scale 0.1697 down to
        # dt 4e-5) -- and any inclined face (fillet, wall) wedges the surface
        # outward into a ridge and folds the corner element.  The window has
        # neither: the normal stays radial and the force is continuous.
        # CAVEAT (measured, mesh A hit 1: 109 kN at w = 0.5 and 129 kN at
        # w = 1.0 against 85 kN for the sharp band): the window acts on a
        # stiff penalty, so a node is still fully loaded where k_n W bite
        # exceeds its yield force, i.e. down to W ~ 0.01 -- the effective
        # edge sits ~4.5 soft_w OUTSIDE the nominal one and moves with k_n.
        # The window is therefore centred 4.5 soft_w inside the nominal edge
        # (SOFT_SHIFT).  Its use: a NARROW taper (0.1-0.2 mm) that makes the
        # sharp band well-posed -- with soft_w = 0 a penetrating node that
        # slides past the edge loses its force at once and Newton can cycle
        # forever at one die position (h20, hit 2 stroke 2, scale 0.1387).
        self.soft_w = float(soft_w)
        if self.soft_w > 0.0 and kind != "flat":
            raise ValueError("the soft edge belongs to the flat profile")
        self.r = float(fillet_r)
        self.theta = onp.radians(float(flank_deg))
        self.height = float(height)
        self.w = float(sigmoid_w)
        if kind not in ("flat", "fillet", "sigmoid"):
            raise ValueError(f"unknown die profile {kind!r}")
        if kind == "fillet" and (self.r <= 0.0 or not 0.0 < self.theta < onp.pi / 2):
            raise ValueError("fillet profile needs r > 0 and 0 < flank < 90 deg")

    @property
    def reach(self):
        """Half-width of the die at its top: candidates beyond it never touch."""
        if self.kind == "flat":
            return self.a + 2.5 * self.soft_w          # window below 1e-3 there
        if self.kind == "fillet":
            t_arc = self.r * onp.sin(self.theta)
            g_arc = self.r * (1.0 - onp.cos(self.theta))
            return self.a + t_arc + max(0.0, self.height - g_arc) / onp.tan(self.theta)
        # sigmoid: the wall is within 1e-3 of the height 15 widths out
        return self.a + 15.0 * self.w

    def recess(self, s):
        """-> (g, dg/ds) at axial offsets s (array); g = inf off the die."""
        s = onp.asarray(s, dtype=float)
        t = onp.abs(s) - self.a
        sg = onp.sign(s)
        g = onp.zeros_like(s)
        dg = onp.zeros_like(s)
        if self.kind == "flat":
            g[t > 2.5 * self.soft_w] = onp.inf
            return g, dg
        if self.kind == "fillet":
            t_arc = self.r * onp.sin(self.theta)
            on_arc = (t > 0.0) & (t <= t_arc)
            root = onp.sqrt(onp.maximum(self.r ** 2 - t[on_arc] ** 2, 1e-12))
            g[on_arc] = self.r - root
            dg[on_arc] = t[on_arc] / root * sg[on_arc]
            on_flank = t > t_arc
            g_arc = self.r * (1.0 - onp.cos(self.theta))
            tan = onp.tan(self.theta)
            g[on_flank] = g_arc + tan * (t[on_flank] - t_arc)
            dg[on_flank] = tan * sg[on_flank]
        else:
            # logistic wall outside the flat: g = H S((t - t0)/w) - g_e with
            # t0 chosen so that g(0) = 0, g_e = H S(-t0/w) = 0.005 mm; the
            # slope at the flat edge is then g_e / w, i.e. smooth, and the
            # wall reaches H/2 at t0 ~ 8.5 w (w = 0.5: 2.5 mm high at
            # t = 3.2 mm, like the Tool2 fillet)
            g_e = 0.005
            t0 = self.w * onp.log(self.height / g_e - 1.0)
            z = onp.clip((t - t0) / self.w, -60.0, 60.0)
            S = 1.0 / (1.0 + onp.exp(-z))
            out = t > 0.0
            g[out] = self.height * S[out] - g_e
            dg[out] = self.height * S[out] * (1.0 - S[out]) / self.w * sg[out]
        off = g > self.height
        g[off] = onp.inf
        dg[off] = 0.0
        return g, dg

    SOFT_SHIFT = 4.5

    def window(self, s):
        """-> (W, dW/ds): stiffness window of the soft edge, 1 and 0 without it."""
        s = onp.asarray(s, dtype=float)
        if self.soft_w <= 0.0:
            return onp.ones_like(s), onp.zeros_like(s)
        # centred SOFT_SHIFT widths INSIDE the nominal edge, so that the
        # effective edge (W ~ 0.01, see the caveat in __init__) sits on it
        z = onp.clip((self.a - self.SOFT_SHIFT * self.soft_w - onp.abs(s)) / self.soft_w,
                     -60.0, 60.0)
        W = 1.0 / (1.0 + onp.exp(-z))
        return W, -onp.sign(s) * W * (1.0 - W) / self.soft_w

    def describe(self):
        if self.kind == "flat" and self.soft_w > 0.0:
            return (f"flat band {2 * self.a:.2f} mm, soft edge: stiffness window "
                    f"of width {self.soft_w:.2f} mm (0.01 at the band edge, 0.5 at "
                    f"{self.SOFT_SHIFT * self.soft_w:.2f} mm inside it)")
        if self.kind == "flat":
            return f"flat band {2 * self.a:.2f} mm, sharp edge"
        if self.kind == "fillet":
            return (f"flat {2 * self.a:.2f} mm + fillet r {self.r:.3f} mm + flank "
                    f"{onp.degrees(self.theta):.0f} deg to {self.height:.1f} mm "
                    f"(half-width at the top {self.reach:.2f} mm)")
        return (f"flat {2 * self.a:.2f} mm + sigmoid wall of width {self.w:.2f} mm "
                f"to {self.height:.1f} mm")


class NodalDieContact:
    """Penalty normal contact + two-direction Coulomb friction at the nodes.

    cand    : indices of the candidate nodes (lateral surface of the bar)
    pts     : reference node coordinates (n_nodes, 3), global frame
    k_n     : normal penalty stiffness, N/mm per node
    mu      : Coulomb friction coefficient (0 = frictionless)
    delta   : slip at which a sticking node reaches the Coulomb limit, mm
    """

    def __init__(self, pts, cand, profile, k_n, mu=0.0, delta=0.01, vec=3, normal="surface"):
        self.pts = onp.asarray(pts, dtype=float)
        self.n = len(self.pts)
        self.cand = onp.asarray(cand, dtype=int)
        self.nc = len(self.cand)
        self.profile = profile
        self.k_n = float(k_n)
        self.mu = float(mu)
        self.delta = float(delta)
        self.vec = int(vec)
        # "surface": the force acts along the true normal of the die face.
        # "radial": the die face only transmits force along the press
        # direction (a bed of vertical pins of height g(s)): the gap is the
        # vertical one, p_y - (y_L - g(s)), and the normal is +-e_y whatever
        # the slope.  With the true normal an inclined face wedges the surface
        # nodes axially outward -- measured: +1.6 mm against +0.5 mm for the
        # node below, the node climbs the wall instead of going down, a ridge
        # forms outside the die and the corner element folds (real fillet in
        # hit 1, sigmoid wall in hit 2, on every mesh).  The radial normal
        # keeps the imprint profile g(s) and the unilateral, continuous force,
        # and drops the wedge.
        if normal not in ("surface", "radial"):
            raise ValueError(f"unknown normal mode {normal!r}")
        self.normal = normal
        # dofs of the 3x3 block of every candidate node, row-major
        d = self.cand[:, None] * self.vec + onp.arange(self.vec)[None, :]      # (nc, 3)
        self.I_ext = onp.repeat(d[:, :, None], self.vec, axis=2).reshape(-1)
        self.J_ext = onp.repeat(d[:, None, :], self.vec, axis=1).reshape(-1)
        self.n_ext = len(self.I_ext)
        self.R = onp.eye(3)
        self.x_mid = 0.0
        self.y_L = -onp.inf
        self.y_R = onp.inf
        self.reset()

    # ---- state ---------------------------------------------------------------
    def reset(self):
        """Start of a stroke: no friction memory, no lagged normal force."""
        self.N = onp.zeros(self.nc)             # lagged normal force (>= 0)
        self.k_t = onp.zeros(self.nc)
        self.Ft_old = onp.zeros((self.nc, 3))   # committed friction force, die frame
        self.u_ref = onp.zeros((self.nc, 3))    # committed displacement, global
        self.active_old = onp.zeros(self.nc, dtype=bool)

    def set_pose(self, R, x_mid, y_L, y_R):
        """Die pose for the step about to be solved: global -> die frame
        rotation R (p_die = R @ p), band centre x_mid (die-frame x), and the
        positions of the lower / upper flat planes."""
        self.R = onp.asarray(R, dtype=float)
        self.x_mid = float(x_mid)
        self.y_L = float(y_L)
        self.y_R = float(y_R)

    # ---- geometry ------------------------------------------------------------
    def _geometry(self, u):
        """-> (d, n, side): signed distance of every candidate to the nearer
        die face (inf off the die), unit face normal pointing into the bar
        (die frame), and side (+1 upper, -1 lower)."""
        p = (self.pts[self.cand] + onp.asarray(u)[self.cand]) @ self.R.T
        s = p[:, 0] - self.x_mid
        g, dg = self.profile.recess(s)
        side = onp.where(p[:, 1] >= 0.5 * (self.y_L + self.y_R), 1.0, -1.0)
        # level set phi >= 0 outside the die; its gradient is the face normal
        phi = onp.where(side > 0, self.y_R + g - p[:, 1], p[:, 1] - self.y_L + g)
        grad = onp.stack([dg, -side, onp.zeros_like(dg)], axis=1)
        if self.normal == "radial":
            # vertical gap; its gradient keeps the slope term, the force
            # direction does not
            n = onp.stack([onp.zeros_like(dg), -side, onp.zeros_like(dg)], axis=1)
            d = phi
            grad_d = grad
        else:
            norm = onp.linalg.norm(grad, axis=1)
            n = grad / norm[:, None]
            d = phi / norm
            grad_d = n                  # first order (curvature dropped)
        ok = onp.isfinite(g)
        d = onp.where(ok, d, onp.inf)
        n = onp.where(ok[:, None], n, 0.0)
        grad_d = onp.where(ok[:, None], grad_d, 0.0)
        return d, n, side, p, grad_d

    # ---- evaluation at a trial displacement ----------------------------------
    def _evaluate(self, u, with_tangent):
        d, n, side, p, grad_d = self._geometry(u)
        pen = onp.where(onp.isfinite(d), onp.maximum(0.0, -d), 0.0)
        active = pen > 0.0
        W, dW = self.profile.window(p[:, 0] - self.x_mid)      # soft edge (1, 0 without)
        Nn = self.k_n * W * pen                                # current normal force
        f = Nn[:, None] * n                                    # die frame
        # friction on the tangent plane, limited by the LAGGED normal force
        P = onp.eye(3)[None] - n[:, :, None] * n[:, None, :]
        du = (onp.asarray(u)[self.cand] - self.u_ref) @ self.R.T
        F_trial = onp.einsum("nij,nj->ni", P, self.Ft_old) - self.k_t[:, None] * onp.einsum("nij,nj->ni", P, du)
        lim = self.mu * self.N
        mag = onp.linalg.norm(F_trial, axis=1)
        fr_on = active & (lim > 0.0)
        slip = fr_on & (mag > lim)
        stick = fr_on & ~slip
        scale = onp.where(slip, lim / onp.maximum(mag, 1e-300), 1.0)
        Ft = onp.where(fr_on[:, None], F_trial * scale[:, None], 0.0)
        f = f + Ft
        K = None
        if with_tangent:
            # K = d f / d u, the derivative of the die force on the node: the
            # penetration grows as the node moves AGAINST n, so the normal
            # block is -k_n n n^T; the caller appends -K to the tangent.
            K = onp.zeros((self.nc, 3, 3))
            K[active] = -(self.k_n * W[active])[:, None, None] * (
                n[active][:, :, None] * grad_d[active][:, None, :])
            # soft edge: the stiffness changes with the axial position
            K[active, :, 0] += (self.k_n * dW[active] * pen[active])[:, None] * n[active]
            if stick.any():
                K[stick] -= self.k_t[stick][:, None, None] * P[stick]
            if slip.any():
                t = F_trial[slip] / mag[slip][:, None]
                K[slip] -= (self.k_t[slip] * lim[slip] / mag[slip])[:, None, None] * (
                    P[slip] - t[:, :, None] * t[:, None, :])
            # global frame: f_g = R^T f, K_g = R^T K R
            K = onp.einsum("ji,njk,kl->nil", self.R, K, self.R)
        f_glob = f @ self.R
        return dict(f=f_glob, f_die=f, K=K, d=d, n=n, side=side, N=Nn, active=active,
                    slip=slip, stick=stick, Ft=Ft, p=p)

    def nodal_terms(self, u, with_tangent):
        """-> (F (n_nodes, 3) force of the die on the nodes, V_ext) with V_ext
        the derivative dF/du as triplet values on (I_ext, J_ext), or None.
        The problem subtracts F from the residual and appends -V_ext."""
        ev = self._evaluate(u, with_tangent)
        F = onp.zeros((self.n, 3))
        F[self.cand] = ev["f"]
        V = ev["K"].reshape(-1) if with_tangent else None
        return F, V

    # ---- between steps ---------------------------------------------------------
    def commit(self, u):
        """Accept the converged state u: friction memory and lagged N."""
        ev = self._evaluate(u, False)
        self.Ft_old = ev["Ft"]
        self.u_ref = onp.asarray(u)[self.cand].copy()
        self.N = ev["N"]
        self.k_t = onp.where(self.N > 0.0, self.mu * self.N / self.delta, 0.0)
        self.active_old = ev["active"]

    def active_sets(self, u):
        """-> (lower node indices, upper node indices) in contact at u."""
        ev = self._evaluate(u, False)
        a = ev["active"]
        return (self.cand[a & (ev["side"] < 0)], self.cand[a & (ev["side"] > 0)])

    def press_forces_kn(self, u):
        """-> (F_lower, F_upper): normal load on each die along the press
        direction at u, kN, both positive."""
        ev = self._evaluate(u, False)
        fy = ev["f_die"][:, 1]
        return (float(fy[ev["side"] < 0].sum() / 1e3), float(-fy[ev["side"] > 0].sum() / 1e3))

    def stats(self, u):
        ev = self._evaluate(u, False)
        a = ev["active"]
        d = ev["d"]
        return dict(n_active=int(a.sum()),
                    n_lower=int((a & (ev["side"] < 0)).sum()),
                    n_upper=int((a & (ev["side"] > 0)).sum()),
                    n_slip=int(ev["slip"].sum()), n_stick=int(ev["stick"].sum()),
                    max_pen_mm=float(-d[a].min()) if a.any() else 0.0,
                    sum_N_kN=float(ev["N"].sum() / 1e3),
                    sum_abs_Ft_kN=float(onp.linalg.norm(ev["Ft"], axis=1).sum() / 1e3),
                    F_lower_kN=float(ev["f_die"][ev["side"] < 0, 1].sum() / 1e3),
                    F_upper_kN=float(-ev["f_die"][ev["side"] > 0, 1].sum() / 1e3))
