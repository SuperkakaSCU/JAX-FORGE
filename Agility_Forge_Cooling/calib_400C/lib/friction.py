"""Nodal Coulomb friction along the bar axis for the press contact.

Why this exists
---------------
Full stick (all three components of a contact node prescribed) shears the
first element layer under the die edge until it degenerates: on the coarse
HEX8 mesh, hit 2 of roll3.1 collapsed a surface element at the end of
stroke 1.  Frictionless axial slip (--axial-slip) removes that, but lets the
material run out from under the die on the free-end side, so the imprint
comes out about 2 mm narrow there and the press load drops by ~15 %.
Real contact lies in between; this adds a finite friction coefficient.

Model
-----
The contact nodes keep their normal and circumferential components
prescribed (as with --axial-slip); their axial (x) component is free and
carries a friction force from the die.  Per node, an elastic-stick /
Coulomb-slip return map, the same structure as rate-independent plasticity:

    F_trial = F_old - k * (u_x - u_x_ref)        k = mu * N / delta
    F       = clip(F_trial, -mu * N, +mu * N)

    F      : axial force of the die on the node (opposes slip)
    N      : normal contact force on the node, lagged by one step (the
             reaction of the last converged state), so the slip limit is
             fixed during a Newton solve
    delta  : the slip at which a sticking node reaches the Coulomb limit
             (a regularisation; small compared with the element size)
    u_x_ref, F_old : the state of the last converged step

The friction force enters the residual as an external nodal force (-F on the
x dof) and its derivative as a diagonal stiffness (+k while sticking, 0 while
sliding), appended to the package's sparse triplets.  mu = 0 reproduces
--axial-slip exactly; large mu approaches full stick in x.

Friction is limited to the axial direction: circumferentially the contact
still sticks (that component is prescribed), as in --axial-slip.
"""
import numpy as onp


class NodalAxialFriction:

    def __init__(self, n_nodes, mu, delta=0.01, vec=3, comp=0):
        self.n = int(n_nodes)
        self.mu = float(mu)
        self.delta = float(delta)
        self.vec = int(vec)
        self.comp = int(comp)
        self.dofs = onp.arange(self.n) * self.vec + self.comp   # x dofs of all nodes
        # triplet extension (diagonal on the x dofs), the interface the
        # problem's _nodal_force_terms expects of every nodal force term
        self.I_ext = self.dofs
        self.J_ext = self.dofs
        self.n_ext = self.n
        self.reset()

    def nodal_terms(self, u, with_tangent):
        """-> (F (n_nodes, 3) force of the die on the nodes, dF/du on
        (I_ext, J_ext) or None); the problem subtracts F and appends -dF."""
        F, dF = self.force_and_stiffness(u)
        Ff = onp.zeros((self.n, self.vec))
        Ff[:, self.comp] = F
        return Ff, (dF if with_tangent else None)

    def reset(self):
        self.active = onp.zeros(self.n, dtype=bool)
        self.N = onp.zeros(self.n)            # lagged normal force (>= 0)
        self.k = onp.zeros(self.n)            # stick stiffness
        self.F_old = onp.zeros(self.n)
        self.u_ref = onp.zeros(self.n)

    # ---- evaluation at a trial displacement ------------------------------
    def force_and_stiffness(self, u):
        """-> (F, dF/du_x) for every node; zero on inactive nodes."""
        ux = onp.asarray(u)[:, self.comp]
        F_trial = self.F_old - self.k * (ux - self.u_ref)
        lim = self.mu * self.N
        F = onp.clip(F_trial, -lim, lim)
        stick = onp.abs(F_trial) < lim
        dF = onp.where(stick, -self.k, 0.0)
        F = onp.where(self.active, F, 0.0)
        dF = onp.where(self.active, dF, 0.0)
        return F, dF

    # ---- between steps -----------------------------------------------------
    def commit(self, u):
        """Accept the converged state u: friction forces and reference slip."""
        F, _ = self.force_and_stiffness(u)
        self.F_old = F
        self.u_ref = onp.asarray(u)[:, self.comp].copy()

    def set_contact(self, nodes, normal_force):
        """New active set and lagged normal forces for the next step.

        nodes        : indices of the contact nodes for the next step
        normal_force : (n_nodes,) normal force magnitude from the last
                       converged reaction (only its values at `nodes` used)
        """
        act = onp.zeros(self.n, dtype=bool)
        act[onp.asarray(nodes, dtype=int)] = True
        N = onp.where(act, onp.abs(onp.asarray(normal_force)), 0.0)
        act &= N > 0.0
        self.active = act
        self.N = N
        self.k = onp.where(act, self.mu * N / self.delta, 0.0)
        lim = self.mu * N
        # a node that left contact carries nothing; one whose normal force
        # dropped keeps at most the new Coulomb limit
        self.F_old = onp.where(act, onp.clip(self.F_old, -lim, lim), 0.0)

    def stats(self, u):
        F, dF = self.force_and_stiffness(u)
        a = self.active
        n_slip = int((a & (dF == 0.0)).sum())
        return dict(n_active=int(a.sum()), n_slip=n_slip,
                    sum_F_kN=float(F.sum() / 1e3),
                    sum_abs_F_kN=float(onp.abs(F).sum() / 1e3),
                    sum_N_kN=float(self.N[a].sum() / 1e3))
