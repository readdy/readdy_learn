import numpy as np

import scipy.integrate as integrate
from scipy import sparse
from scipy.sparse import linalg as splin
import itertools
import matplotlib.pyplot as plt


def ld_derivative2(data, xs, alpha, maxit=1000, verbose=False):
    assert isinstance(data, np.ndarray)
    data = data.squeeze()
    assert len(data.shape) == 1

    epsilon = 1e-8

    n = len(data)

    # require f(0) = 0
    data = data - data[0]

    A = lambda v: integrate.cumtrapz(y=v, x=xs[:len(v)], initial=0)
    AT = lambda w: np.trapz(y=w, x=xs[:len(w)]) * np.ones(len(w)) - A(w)

    u = np.gradient(data, xs)


def ld_derivative(data, timestep, alpha, maxit=1000, verbose=False):
    assert isinstance(data, np.ndarray)
    data = data.squeeze()
    assert len(data.shape) == 1

    epsilon = 1e-8

    n = len(data)

    # require f(0) = 0
    data = data - data[0]

    # Construct antidifferentiation operator and its adjoint.
    ATT = lambda w: (sum(w) * np.ones(n + 1) - np.transpose(
        np.concatenate(([sum(w) / 2.0], np.cumsum(w) - w / 2.0)))) * timestep
    A = lambda v: np.cumsum(v)
    AT = lambda w: (sum(w) * np.ones(len(w)) - np.transpose(np.concatenate(([0.0], np.cumsum(w[:-1])))))
    # Construct differentiation matrix.
    c = np.ones(n)
    D = sparse.spdiags([-c, c], [0, 1], n, n) / timestep
    mask = np.ones((n, n))
    mask[-1, -1] = 0.0
    D = sparse.dia_matrix(D.multiply(mask))
    DT = D.transpose()

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


def fd_coeff(xbar, x, k=1):
    n = len(x)
    if k >= n:
        raise ValueError("*** length(y) = {} must be larger than k = {}".format(len(x), k))

    # change to m = n - 1 if you want to compute coefficients for all
    # possible derivatives.Then modify to output all of C.
    m = k

    c1 = 1
    c4 = x[0] - xbar
    C = np.zeros(shape=(n, m + 1))
    C[0, 0] = 1
    for i in range(1, n):
        mn = min(i, m)
        c2 = 1
        c5 = c4
        c4 = x[i] - xbar
        for j in range(0, i):
            c3 = x[i] - x[j]
            c2 = c2 * c3
            if j == i - 1:
                for s in range(mn, 0, -1):
                    C[i, s] = c1 * (s * C[i - 1, s - 1] - c5 * C[i - 1, s]) / c2
                C[i, 0] = -c1 * c5 * C[i - 1, 0] / c2
            for s in range(mn, 0, -1):
                C[j, s] = (c4 * C[j, s] - s * C[j, s - 1]) / c3
            C[j, 0] = c4 * C[j, 0] / c3
        c1 = c2
    c = C[:, -1]
    return c


from collections import deque

def window(seq, n=2):
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win


if __name__ == '__main__':
    x0 = np.arange(0, 2.0 * np.pi, 0.05)

    # xpts = np.array([.5, .75, 1., 1.25])
    # print(fd_coeff(1, xpts, k=1))
    #  expect 0.66667  -4.00000   2.00000   1.33333

    testf = np.array([np.sin(x) for x in x0])
    true_deriv = [np.cos(x) for x in x0]

    deriv = [fd_coeff(0, x0[0:2], k=1).dot(testf[0:2])]

    for w in window(x0, n=3):
        print(w)

    for ix, w in enumerate(window(x0, n=5)):
        x = x0[ix]
        y = testf[ix:ix + len(w)]
        coeff = fd_coeff(x, w, k=1)
        deriv.append(coeff.dot(y))

    plt.plot(x0[1:-2], np.array(deriv))
    plt.show()
    # deriv = np.array([ for xbar in x0])
    # plt.plot(deriv)
    # plt.show()
    # print(slope)
    # print(np.cos(xbar))

    # testf = testf + np.random.normal(0.0, 0.04, x0.shape)
    #
    # deriv_sm = ld_derivative(testf, timestep=0.05, alpha=5e-2, verbose=True)
    # deriv_lrg = ld_derivative(testf, timestep=0.05, alpha=1e-1)
    #
    #
    # plt.plot(testf, label='fun')
    # plt.plot(deriv_sm, label='alpha=5e-2')
    # plt.plot(deriv_lrg, label='alpha=1e-1')
    # plt.plot(true_deriv, label='derivative')
    # plt.legend()
    # plt.show()