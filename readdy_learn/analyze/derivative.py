import numpy as np

from scipy import sparse
from scipy.sparse import linalg as splin

import matplotlib.pyplot as plt


def ld_derivative(data, timestep, alpha, maxit=1000, verbose=False):
    assert isinstance(data, np.ndarray)
    data = data.squeeze()
    assert len(data.shape) == 1

    epsilon = 1e-8

    n = len(data)

    # require f(0) = 0
    data = data - data[0]

    # Construct antidifferentiation operator and its adjoint.
    A = lambda v: np.cumsum(v)
    AT = lambda w: (sum(w) * np.ones(len(w)) - np.transpose(np.concatenate(([0.0], np.cumsum(w[:-1])))))
    # Construct differentiation matrix.
    c = np.ones(n)
    D = sparse.spdiags([-c, c], [0, 1], n, n) / timestep
    mask = np.ones((n, n))
    mask[-1, -1] = 0.0
    D = sparse.dia_matrix(D.multiply(mask))
    DT = D.transpose()
    # Since Au( 0 ) = 0, we need to adjust.
    data = data - data[0]
    # Default initialization is naive derivative.
    u = np.concatenate(([0], np.diff(data)))
    # Precompute.
    ATd = AT(data)

    # Main loop.
    for ii in range(1, maxit + 1):
        # Diagonal matrix of weights, for linearizing E-L equation.
        Q = sparse.spdiags(1. / np.sqrt((D * u) ** 2.0 + epsilon), 0, n, n)
        # Linearized diffusion matrix, also approximation of Hessian.
        L = DT * Q * D
        # Gradient of functional.
        g = AT(A(u)) - ATd
        g = g + alpha * L * u
        # Build preconditioner.
        c = np.cumsum(range(n, 0, -1))
        B = alpha * L + sparse.spdiags(c[::-1], 0, n, n)
        # droptol = 1.0e-2
        R = sparse.dia_matrix(np.linalg.cholesky(B.todense()))
        # Prepare to solve linear equation.
        tol = 1.0e-4
        maxit = None

        linop = lambda v: (alpha * L * v + AT(A(v)))
        linop = splin.LinearOperator((n, n), linop)

        [s, info_i] = splin.cg(A=linop, b=-g, x0=None, tol=tol, maxiter=maxit, xtype=None, M=np.dot(R.transpose(), R))
        if verbose:
            print('iteration {0:4d}: relative change = {1:.3e}, gradient norm = {2:.3e}'
                  .format(ii, np.linalg.norm(s[0]) / np.linalg.norm(u), np.linalg.norm(g)))
            if info_i > 0:
                print("WARNING - convergence to tolerance not achieved!")
            elif info_i < 0:
                print("WARNING - illegal input or breakdown")

        # Update current solution
        u = u + s

    return u / timestep

if __name__ == '__main__':
    x0 = np.arange(0, 2.0 * np.pi, 0.05)

    testf = [np.sin(x) for x in x0]
    true_deriv = [np.cos(x) for x in x0]

    testf = testf + np.random.normal(0.0, 0.04, x0.shape)

    deriv_sm = ld_derivative(testf, timestep=0.05, alpha=5e-2, verbose=True)
    deriv_lrg = ld_derivative(testf, timestep=0.05, alpha=1e-1)

    plt.plot(testf, label='fun')
    plt.plot(deriv_sm, label='alpha=5e-2')
    plt.plot(deriv_lrg, label='alpha=1e-1')
    plt.plot(true_deriv, label='derivative')
    plt.legend()
    plt.show()
