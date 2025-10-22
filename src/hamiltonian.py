import numpy as np
import netket as nk

def build_hamiltonian(L, J, A, D, Bz, delta):

    g  = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
    ham = nk.operator.LocalOperator(hi, dtype=complex)

    # Disorder strengths (std. dev.)
    sigma_J   = delta * J   # fluctuations around J
    sigma_A   = delta * A   # fluctuations around A
    sigma_D   = delta * D   # fluctuations around D
    #sigma_Bz  = 0.2 * Bz  # fluctuations around Bz

    # Pre-draw all disorder from N(0,1)
    rho_J   = { (i,j): np.random.normal() for i,j in g.edges() }
    rho_A   = { (i,j): np.random.normal() for i,j in g.edges() }
    rho_D   = { (i,j): np.random.normal() for i,j in g.edges() }
    #eta_Bz  = {  i   : np.random.normal() for i   in range(g.n_nodes) }

    # Build the Hamiltonian
    for (i,j) in g.edges():
        Jij = J   + sigma_J  * rho_J[(i,j)]
        Aij = A   + sigma_A  * rho_A[(i,j)]
        Dij = D   + sigma_D  * rho_D[(i,j)]

        # XX + YY
        ham += -Jij * nk.operator.spin.sigmax(hi,i)*nk.operator.spin.sigmax(hi,j)
        ham += -Jij * nk.operator.spin.sigmay(hi,i)*nk.operator.spin.sigmay(hi,j)
        # ZZ
        ham += -Aij * nk.operator.spin.sigmaz(hi,i)*nk.operator.spin.sigmaz(hi,j)
        # DMI
        pos_i, pos_j = np.array(g.positions[i]), np.array(g.positions[j])
        u_ij = np.append((pos_j-pos_i)/np.linalg.norm(pos_j-pos_i), 0)
        cross = np.cross(u_ij, [0,0,1])
        Dx = (nk.operator.spin.sigmay(hi,i)*nk.operator.spin.sigmaz(hi,j)
            -nk.operator.spin.sigmaz(hi,i)*nk.operator.spin.sigmay(hi,j))
        Dy = (nk.operator.spin.sigmaz(hi,i)*nk.operator.spin.sigmax(hi,j)
            -nk.operator.spin.sigmax(hi,i)*nk.operator.spin.sigmaz(hi,j))
        Dz = (nk.operator.spin.sigmax(hi,i)*nk.operator.spin.sigmay(hi,j)
            -nk.operator.spin.sigmay(hi,i)*nk.operator.spin.sigmax(hi,j))
        D_vec = np.array([Dx, Dy, Dz])
        ham += -Dij*(cross[0]*D_vec[0] + cross[1]*D_vec[1] + cross[2]*D_vec[2])
    ''' 
        # Applying uniform Bz field
    for i in range(g.n_nodes):
        ham += Bz * nk.operator.spin.sigmaz(hi, i)
    '''

    # Apply Bz only at boundary nodes
    for i in range(g.n_nodes):
        x, y = g.positions[i]
        if x == 0 or x == L-1 or y == 0 or y == L-1:
            ham += Bz * nk.operator.spin.sigmaz(hi, i)

    '''
    # Disordered boundary field
    for i in range(g.n_nodes):
        x, y = g.positions[i]
        if x in (0, L-1) or y in (0, L-1):
            Bzi = Bz + sigma_Bz * eta_Bz[i]
            ham += Bzi * nk.operator.spin.sigmaz(hi, i)
    '''

    return ham, hi, g