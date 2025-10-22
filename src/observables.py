import numpy as np
import netket as nk
from netket.experimental.observable import Renyi2EntanglementEntropy

def compute_spin_expectations(vstate, hi, g):
    """Compute <Sx>, <Sy>, <Sz> and errors at each site."""
    sx, sy, sz = [], [], []
    sx_err, sy_err, sz_err = [], [], []

    vstate.sample(n_samples=1000)

    for i in range(g.n_nodes):
        ex = vstate.expect(nk.operator.spin.sigmax(hi, i))
        ey = vstate.expect(nk.operator.spin.sigmay(hi, i))
        ez = vstate.expect(nk.operator.spin.sigmaz(hi, i))

        sx.append(ex.mean.real / 2)
        sy.append(ey.mean.real / 2)
        sz.append(ez.mean.real / 2)

        sx_err.append(ex.error_of_mean.real / 2)
        sy_err.append(ey.error_of_mean.real / 2)
        sz_err.append(ez.error_of_mean.real / 2)

    return np.array(sx), np.array(sy), np.array(sz), np.array(sx_err), np.array(sy_err), np.array(sz_err)


def compute_skyrmion_number(sx, sy, sz, L, g):
    """
    Compute Skyrmion density, Skyrmion number (C), and Skyrmion stability (Q).

    """
    spin_vectors = np.stack([sx, sy, sz], axis=1)
    norms = np.linalg.norm(spin_vectors, axis=1, keepdims=True)
    n_vectors = spin_vectors / norms

    triangles = []
    for x in range(L - 1):
        for y in range(L - 1):
            i = x * L + y
            j = (x + 1) * L + y
            k = x * L + (y + 1)
            l = (x + 1) * L + (y + 1)
            triangles.append((i, j, l))
            triangles.append((i, l, k))

    sky_density = []
    q_density = []
    C = 0.0
    Q = 0.0
    for i, j, k in triangles:
        ni = n_vectors[i]
        mi = 2*spin_vectors[i]
        nj = n_vectors[j]
        mj = 2*spin_vectors[j]
        nk_ = n_vectors[k]
        mk_ = 2*spin_vectors[k]
        numerator = np.dot(ni, np.cross(nj, nk_))
        num = np.dot(mi, np.cross(mj, mk_))
        denominator = 1 + np.dot(ni, nj) + np.dot(nj, nk_) + np.dot(nk_, ni)
        den = 1 + np.dot(mi, mj) + np.dot(mj, mk_) + np.dot(mk_, mi)
        angle = np.arctan2(numerator, denominator)
        ang = np.arctan2(num, den)
        sky_density.append(angle)
        q_density.append(ang)
        C += angle
        Q += ang
    C /= 2 * np.pi
    Q /= 2 * np.pi
    sky_density = np.array(sky_density)
    q_density = np.array(q_density)

    return float(C), float(Q), sky_density, q_density, triangles



def compute_renyi_entropies(vstate, hi, g, L):
    """Compute site-wise Renyi-2 entropies."""
    entropies, errors = [], []
    for i in range(g.n_nodes):
        obs = Renyi2EntanglementEntropy(hi, [i])
        stats = vstate.expect(obs)
        entropies.append(stats.mean)
        errors.append(stats.error_of_mean)
    renyi = np.array(entropies)
    renyi_err = np.array(errors)
    entropies = np.array(entropies).reshape((L, L))
    return entropies, renyi, renyi_err


def compute_topological_entropy(vstate, hi, g, L):
    """
    Compute Topological Entanglement Entropy (TEE) using the Kitaev-Preskill method:
    S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
    """
    positions = np.array(g.positions)

    # Only interior sites (avoid boundary)
    interior = [
        i for i, (x, y) in enumerate(positions)
        if 0 < x < L - 1 and 0 < y < L - 1
    ]

    # Center of lattice
    cx, cy = (L - 1) / 2, (L - 1) / 2

    # Partition into angular sectors
    angles = {}
    for i in interior:
        x, y = positions[i]
        theta = np.arctan2(y - cy, x - cx) % (2 * np.pi)
        angles[i] = theta

    sector_width = 2 * np.pi / 3
    A = [i for i, theta in angles.items() if 0 <= theta < sector_width]
    B = [i for i, theta in angles.items() if sector_width <= theta < 2 * sector_width]
    C = [i for i, theta in angles.items() if 2 * sector_width <= theta < 2 * np.pi]

    AB, BC, AC = list(set(A) | set(B)), list(set(B) | set(C)), list(set(A) | set(C))
    ABC = list(set(A) | set(B) | set(C))

    def renyi2(region):
        obs = Renyi2EntanglementEntropy(hi, region)
        stats = vstate.expect(obs)
        return stats.mean, stats.error_of_mean

    Ss = {}
    for name, region in [("A", A), ("B", B), ("C", C),
                         ("AB", AB), ("BC", BC), ("AC", AC), ("ABC", ABC)]:
        mean, err = renyi2(region)
        Ss[name] = (mean, err)

    Stopo = (Ss["A"][0] + Ss["B"][0] + Ss["C"][0]
             - Ss["AB"][0] - Ss["BC"][0] - Ss["AC"][0]
             + Ss["ABC"][0])
    err = np.sqrt(sum(e[1]**2 for e in Ss.values()))

    return float(Stopo), float(err)