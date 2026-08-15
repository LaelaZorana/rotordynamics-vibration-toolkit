"""Rotor dynamics: Jeffcott rotor and a Rayleigh beam finite element shaft
model with lumped disks, isotropic bearings and gyroscopic effects."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eig, eigh

from .beam import circular_section


# ----------------------------------------------------------------------------
# Jeffcott rotor
# ----------------------------------------------------------------------------
@dataclass
class Jeffcott:
    """Massless flexible shaft, central disk of mass m, eccentricity e, damping c."""

    m: float
    k: float
    c: float = 0.0
    e: float = 0.0

    @property
    def omega_cr(self) -> float:
        """Critical speed in rad/s, equal to sqrt(k/m)."""
        return float(np.sqrt(self.k / self.m))

    @property
    def rpm_cr(self) -> float:
        return self.omega_cr * 60.0 / (2.0 * np.pi)

    @property
    def zeta(self) -> float:
        return self.c / (2.0 * np.sqrt(self.k * self.m))

    def response(self, Omega):
        """Whirl radius and phase lag (rad) of the disk centre versus spin speed."""
        Omega = np.asarray(Omega, dtype=float)
        r = Omega / self.omega_cr
        z = self.zeta
        amp = self.e * r**2 / np.sqrt((1.0 - r**2) ** 2 + (2.0 * z * r) ** 2)
        phase = np.arctan2(2.0 * z * r, 1.0 - r**2)
        return amp, phase


# ----------------------------------------------------------------------------
# Finite element shaft model
# ----------------------------------------------------------------------------
def _beam_element_matrices(E, I, rho, A, L):
    """Planar Euler-Bernoulli stiffness, translational and rotary mass matrices.

    Dofs: (v1, v1', v2, v2'). Rotary mass uses rho*I (Rayleigh beam)."""
    Ke = E * I / L**3 * np.array(
        [
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L**2, -6 * L, 2 * L**2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L**2, -6 * L, 4 * L**2],
        ]
    )
    Mt = rho * A * L / 420.0 * np.array(
        [
            [156, 22 * L, 54, -13 * L],
            [22 * L, 4 * L**2, 13 * L, -3 * L**2],
            [54, 13 * L, 156, -22 * L],
            [-13 * L, -3 * L**2, -22 * L, 4 * L**2],
        ]
    )
    Mr = rho * I / (30.0 * L) * np.array(
        [
            [36, 3 * L, -36, 3 * L],
            [3 * L, 4 * L**2, -3 * L, -(L**2)],
            [-36, -3 * L, 36, -3 * L],
            [3 * L, -(L**2), -3 * L, 4 * L**2],
        ]
    )
    return Ke, Mt, Mr


@dataclass
class Disk:
    node: int
    m: float
    Id: float  # diametral mass moment of inertia
    Ip: float  # polar mass moment of inertia


@dataclass
class Bearing:
    node: int
    k: float
    c: float = 0.0


@dataclass
class ShaftFE:
    """Uniform circular shaft discretised into n_el Rayleigh beam elements.

    Global dof ordering per node: [x, x', y, y'] where the prime is the slope
    d/dz. Rotation about x is minus y' and rotation about y is x'."""

    L: float
    d: float
    E: float
    rho: float
    n_el: int = 10
    disks: list = field(default_factory=list)
    bearings: list = field(default_factory=list)

    def __post_init__(self):
        self.A, self.I = circular_section(self.d)
        self.n_nodes = self.n_el + 1
        self.ndof = 4 * self.n_nodes
        self.z = np.linspace(0.0, self.L, self.n_nodes)
        self._assemble()

    # dof helpers -----------------------------------------------------------
    @staticmethod
    def dof(node, kind):
        """kind: 'x','xs','y','ys' (s = slope)."""
        return 4 * node + {"x": 0, "xs": 1, "y": 2, "ys": 3}[kind]

    def add_disk(self, node, m, Id, Ip):
        self.disks.append(Disk(node, m, Id, Ip))
        self._assemble()

    def add_bearing(self, node, k, c=0.0):
        self.bearings.append(Bearing(node, k, c))
        self._assemble()

    # assembly --------------------------------------------------------------
    def _assemble(self):
        n = self.ndof
        M = np.zeros((n, n))
        K = np.zeros((n, n))
        G = np.zeros((n, n))
        C = np.zeros((n, n))
        Le = self.L / self.n_el
        Ke, Mt, Mr = _beam_element_matrices(self.E, self.I, self.rho, self.A, Le)
        for e in range(self.n_el):
            ix = [self.dof(e, "x"), self.dof(e, "xs"), self.dof(e + 1, "x"), self.dof(e + 1, "xs")]
            iy = [self.dof(e, "y"), self.dof(e, "ys"), self.dof(e + 1, "y"), self.dof(e + 1, "ys")]
            for idx in (ix, iy):
                K[np.ix_(idx, idx)] += Ke
                M[np.ix_(idx, idx)] += Mt + Mr
            # shaft gyroscopic coupling: polar inertia per length is 2 rho I
            G[np.ix_(ix, iy)] += 2.0 * Mr
            G[np.ix_(iy, ix)] -= 2.0 * Mr
        for dsk in self.disks:
            for kind in ("x", "y"):
                M[self.dof(dsk.node, kind), self.dof(dsk.node, kind)] += dsk.m
            for kind in ("xs", "ys"):
                M[self.dof(dsk.node, kind), self.dof(dsk.node, kind)] += dsk.Id
            a, b = self.dof(dsk.node, "xs"), self.dof(dsk.node, "ys")
            G[a, b] += dsk.Ip
            G[b, a] -= dsk.Ip
        for brg in self.bearings:
            for kind in ("x", "y"):
                i = self.dof(brg.node, kind)
                K[i, i] += brg.k
                C[i, i] += brg.c
        self.M, self.K, self.G, self.C = M, K, G, C

    # analysis ---------------------------------------------------------------
    def undamped_frequencies(self, n_modes=6, fixed_dofs=None):
        """Non rotating natural frequencies (rad/s) with optional pinned dofs."""
        keep = np.arange(self.ndof)
        if fixed_dofs is not None:
            keep = np.setdiff1d(keep, np.asarray(fixed_dofs))
        Kr = self.K[np.ix_(keep, keep)]
        Mr = self.M[np.ix_(keep, keep)]
        lam, phi = eigh(Kr, Mr)
        lam = np.clip(lam, 0.0, None)
        full = np.zeros((self.ndof, phi.shape[1]))
        full[keep] = phi
        return np.sqrt(lam[:n_modes]), full[:, :n_modes]

    def pinned_end_dofs(self):
        """Dofs to fix for a simply supported shaft (translations at both ends)."""
        last = self.n_nodes - 1
        return [self.dof(0, "x"), self.dof(0, "y"), self.dof(last, "x"), self.dof(last, "y")]

    def whirl_frequencies(self, Omega, n_modes=6):
        """Damped whirl frequencies (rad/s, positive) at spin speed Omega (rad/s).

        Returns (omega, direction) with direction +1 forward, -1 backward, sorted
        by frequency. Uses the state space eigenproblem with C + Omega G."""
        n = self.ndof
        Minv = np.linalg.inv(self.M)
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -Minv @ self.K
        A[n:, n:] = -Minv @ (self.C + Omega * self.G)
        lam, vec = eig(A)
        w = lam.imag
        mask = w > 1e-6
        w = w[mask]
        vec = vec[:n, mask]
        order = np.argsort(w)
        w, vec = w[order], vec[:, order]
        direction = np.zeros(w.size)
        for i in range(w.size):
            v = vec[:, i]
            x = v[0::4]
            y = v[2::4]
            j = np.argmax(np.abs(x) ** 2 + np.abs(y) ** 2)
            s = np.imag(np.conj(x[j]) * y[j])
            direction[i] = -1.0 if s > 0 else 1.0
        return w[:n_modes], direction[:n_modes]

    def campbell(self, Omega_range, n_modes=6):
        """Whirl frequency map over spin speeds. Returns (W, D) arrays of shape
        (len(Omega_range), n_modes)."""
        W = np.zeros((len(Omega_range), n_modes))
        D = np.zeros_like(W)
        for i, Om in enumerate(Omega_range):
            w, d = self.whirl_frequencies(Om, n_modes)
            W[i, : w.size] = w
            D[i, : d.size] = d
        return W, D

    def critical_speeds(self, Omega_range, n_modes=6):
        """Synchronous critical speeds by detecting crossings of omega = Omega
        on the Campbell diagram. Returns list of (Omega_cr, direction)."""
        Omega_range = np.asarray(Omega_range, dtype=float)
        W, D = self.campbell(Omega_range, n_modes)
        crits = []
        for k in range(n_modes):
            g = W[:, k] - Omega_range
            for i in range(len(Omega_range) - 1):
                if g[i] > 0 and g[i + 1] <= 0:
                    Om = Omega_range[i] - g[i] * (Omega_range[i + 1] - Omega_range[i]) / (g[i + 1] - g[i])
                    crits.append((float(Om), float(D[i, k])))
        crits.sort()
        return crits

    def unbalance_response(self, Omega_range, node, me, n_modes=None):
        """Steady state amplitude at each node for unbalance me (kg m) at node.

        Solves (K - Omega^2 M + i Omega (C + Omega G)) q = f with a rotating
        force in the x, y plane. Returns array (len(Omega), n_nodes) of radii."""
        Omega_range = np.asarray(Omega_range, dtype=float)
        out = np.zeros((Omega_range.size, self.n_nodes))
        for i, Om in enumerate(Omega_range):
            f = np.zeros(self.ndof, dtype=complex)
            f[self.dof(node, "x")] = me * Om**2
            f[self.dof(node, "y")] = -1j * me * Om**2
            Z = self.K - Om**2 * self.M + 1j * Om * (self.C + Om * self.G)
            q = np.linalg.solve(Z, f)
            out[i] = np.abs(q[0::4])
        return out
