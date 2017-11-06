import numpy as np

import scipy.integrate as integrate
from scipy import sparse
from scipy.sparse import linalg as splin
import collections
import matplotlib.pyplot as plt


def approx_jacobian(x, func, epsilon, *args):
    """
    Approximate the Jacobian matrix of a callable function.

    Parameters
    ----------
    x : array_like
        The state vector at which to compute the Jacobian matrix.
    func : callable f(x,*args)
        The vector-valued function.
    epsilon : float
        The perturbation used to determine the partial derivatives.
    args : sequence
        Additional arguments passed to func.

    Returns
    -------
    An array of dimensions ``(lenf, lenx)`` where ``lenf`` is the length
    of the outputs of `func`, and ``lenx`` is the number of elements in
    `x`.

    Notes
    -----
    The approximation is done using forward differences.

    """
    x0 = np.asfarray(x)
    f0 = np.atleast_1d(func(*((x0,) + args)))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon
        jac[i] = (func(*((x0 + dx,) + args)) - f0) / epsilon
        dx[i] = 0.0

    return jac.transpose()


def fd_coeff(xbar, x, k=1):
    n = len(x)
    if k >= n:
        raise ValueError("*** length(y) = {} must be larger than k = {}".format(len(x), k))

    # change to m = n - 1 to compute coeffs for all derivatives, then output C
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


def window_evensteven(seq, width=1):
    it = iter(seq)
    win = collections.deque((next(it, None) for _ in range(1 + width)), maxlen=2 * width + 1)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win
    while len(win) > width + 1:
        win.popleft()
        yield win


def get_differentiation_operator(xs, chopright=False):
    n_nodes = len(xs)
    wwidth = 1

    if chopright:
        roffset = 2
    else:
        roffset = 0

    D_data = np.empty(shape=(3 * n_nodes - 2 - roffset,))
    D_row_data = np.empty_like(D_data)
    D_col_data = np.empty_like(D_data)

    offset = 0
    for ix, wx in enumerate(window_evensteven(xs, width=wwidth)):
        if chopright and ix == len(xs)-1:
            continue
        x = xs[ix]
        coeff = fd_coeff(x, wx, k=1)
        D_data[offset:offset + len(coeff)] = coeff
        D_row_data[offset:offset + len(coeff)] = np.ones(shape=(len(coeff),)) * ix
        D_col_data[offset:offset + len(coeff)] = np.arange(max((0, ix - wwidth)), min(ix + wwidth + 1, n_nodes))
        offset += len(coeff)

    if chopright:
        return sparse.bsr_matrix((D_data, (D_row_data, D_col_data)), shape=(n_nodes-1, n_nodes))
    else:
        return sparse.bsr_matrix((D_data, (D_row_data, D_col_data)), shape=(n_nodes, n_nodes))


def get_differentiation_operator_midpoint(xs):
    n_nodes = len(xs)
    D_data = np.empty(shape=(2 * n_nodes - 2,))
    D_row_data = np.empty_like(D_data)
    D_col_data = np.empty_like(D_data)

    dx = np.diff(xs)

    for ix in range(n_nodes - 1):
        iix1 = 2 * ix
        iix2 = 2 * ix + 1
        d = dx[ix]
        D_data[iix1] = -1. / d
        D_data[iix2] = 1. / d
        D_row_data[iix1] = ix
        D_row_data[iix2] = ix
        D_col_data[iix1] = ix
        D_col_data[iix2] = ix + 1

    # mid_xs = .5 * (xs[:-1] + xs[1:])
    return sparse.bsr_matrix((D_data, (D_row_data, D_col_data)), shape=(n_nodes - 1, n_nodes))


def ld_derivative(data, xs, alpha, maxit=1000, linalg_solver_maxit=10000, tol=1e-8, verbose=False):
    assert isinstance(data, np.ndarray)
    data = data.squeeze()
    assert len(data.shape) == 1

    epsilon = 1e-8

    n = len(data)

    # require f(0) = 0
    data = data - data[0]

    # differentiation operator
    D = get_differentiation_operator_midpoint(xs)
    D_T = D.transpose()

    def A(v):
        # integrate v from 0 to x
        return integrate.cumtrapz(y=v, x=xs, initial=0)

    def A_adjoint(w):
        # integrate w from x to L <=> int_0^Lw - int_0^xw
        full_integral = integrate.trapz(w, xs)
        # assert np.isclose(full_integral, A(w)[-1]), "got {}, but expected {}".format(full_integral, A(w)[-1])
        return np.ones_like(w) * full_integral - integrate.cumtrapz(w, xs, initial=0)

    AAstar = lambda v: A_adjoint(A(v))

    u = np.gradient(data, xs)

    KT_data = A_adjoint(data)

    xs_diff = np.diff(xs)
    # DX = sparse.spdiags(np.diff(xs), 0, n - 1, n - 1)

    E_n = sparse.dia_matrix((n - 1, n - 1), dtype=xs.dtype)

    # Main loop.
    for ii in range(1, maxit + 1):
        # Diagonal matrix of weights, for linearizing E-L equation.
        E_n.setdiag(xs_diff * (1. / np.sqrt(np.diff(u) ** 2.0 + epsilon))) #np.diff(u)
        # sparse.spdiags(1. / np.sqrt(np.diff(u) ** 2.0 + epsilon), 0, n - 1, n - 1)
        Q = E_n #DX *
        L = D_T * Q * D
        # Gradient of functional.
        g = AAstar(u) - KT_data
        g = g + alpha * L * u
        # Build preconditioner.
        c = np.cumsum(range(n, 0, -1))
        B = alpha * L + sparse.spdiags(c[::-1], 0, n, n)
        # droptol = 1.0e-2
        R = sparse.dia_matrix(np.linalg.cholesky(B.todense()))
        # Prepare to solve linear equation.
        linop = lambda v: (alpha * L * v + AAstar(v))
        linop = splin.LinearOperator((n, n), linop)

        [s, info_i] = splin.bicgstab(A=linop, b=-g, x0=None, tol=tol, maxiter=linalg_solver_maxit,
                                     xtype=None, M=np.dot(R.transpose(), R))
        if verbose:
            print('iteration {0:4d}: relative change = {1:.3e}, gradient norm = {2:.3e}'
                  .format(ii, np.linalg.norm(s[0]) / np.linalg.norm(u), np.linalg.norm(g)))
            if info_i > 0:
                print("WARNING - convergence to tolerance not achieved!")
            elif info_i < 0:
                print("WARNING - illegal input or breakdown")

        # Update current solution
        u = u + s

    return u


def test_finite_differences():
    x0 = np.arange(0, 2.0 * np.pi, 0.05)
    print(len(x0))
    xx = []
    for x in x0:
        if np.random.random() < .2:
            xx.append(x)
    x0 = np.array(xx)

    print(len(x0))
    testf = np.array([np.sin(x) for x in x0])
    true_deriv = [np.cos(x) for x in x0]

    wwidth = 5
    deriv = np.empty_like(x0)
    for ix, (wx, wy) in enumerate(zip(window_evensteven(x0, width=wwidth), window_evensteven(testf, width=wwidth))):
        x = x0[ix]
        coeff = fd_coeff(x, wx, k=1)
        deriv[ix] = coeff.dot(wy)
    plt.plot(x0, deriv, 'o')
    plt.plot(x0, true_deriv)
    plt.show()
    print(np.array(deriv) - np.array(true_deriv))


def test_ld_derivative():
    x0 = np.arange(0, 2.0 * np.pi, 0.05)
    xx = []
    for x in x0:
        if np.random.random() < .5 :
            xx.append(x)
    x0 = np.array(xx)

    testf = np.array([np.sin(x) for x in x0])
    testf = testf + np.random.normal(0.0, 0.04, x0.shape)
    true_deriv = [np.cos(x) for x in x0]

    D = get_differentiation_operator(x0)
    get_differentiation_operator_midpoint(x0)
    deriv = D * testf

    ld_deriv = ld_derivative(testf, x0, alpha=5e-4, verbose=True)

    plt.plot(testf, label='f')
    plt.plot(true_deriv, label='df')
    plt.plot(deriv, label='approx df')
    # plt.plot(dmidxs, Dmidderiv)
    plt.plot(ld_deriv, label='total variation df')
    plt.legend()
    plt.show()


    # deriv_sm = ld_derivative(testf, timestep=0.05, alpha=5e-4, verbose=False)
    # deriv_lrg = ld_derivative(testf, timestep=0.05, alpha=1e-1)

    # plt.plot(testf, label='fun')
    # plt.plot(deriv_sm, label='alpha=5e-4')
    # plt.plot(deriv_lrg, label='alpha=1e-1')
    # plt.plot(true_deriv, label='derivative')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    # test_finite_differences()
    test_ld_derivative()



#
# def ld_derivative(data, timestep, alpha, maxit=1000, verbose=False):
#     assert isinstance(data, np.ndarray)
#     data = data.squeeze()
#     assert len(data.shape) == 1
#
#     epsilon = 1e-8
#
#     n = len(data)
#
#     # require f(0) = 0
#     data = data - data[0]
#
#     # Construct antidifferentiation operator and its adjoint.
#     A = lambda v: np.cumsum(v)
#     AT = lambda w: (sum(w) * np.ones(len(w)) - np.transpose(np.concatenate(([0.0], np.cumsum(w[:-1])))))
#     # Construct differentiation matrix.
#     c = np.ones(n)
#     D = sparse.spdiags([-c, c], [0, 1], n, n) / timestep
#     mask = np.ones((n, n))
#     mask[-1, -1] = 0.0
#     D = sparse.dia_matrix(D.multiply(mask))
#     DT = D.transpose()
#
#     # Default initialization is naive derivative.
#     u = np.concatenate(([0], np.diff(data)))
#     # Precompute.
#     ATd = AT(data)
#
#     # Main loop.
#     for ii in range(1, maxit + 1):
#         # Diagonal matrix of weights, for linearizing E-L equation.
#         Q = sparse.spdiags(1. / np.sqrt((D * u) ** 2.0 + epsilon), 0, n, n)
#         # Linearized diffusion matrix, also approximation of Hessian.
#         L = DT * Q * D
#         # Gradient of functional.
#         g = AT(A(u)) - ATd
#         g = g + alpha * L * u
#         # Build preconditioner.
#         c = np.cumsum(range(n, 0, -1))
#         B = alpha * L + sparse.spdiags(c[::-1], 0, n, n)
#         # droptol = 1.0e-2
#         R = sparse.dia_matrix(np.linalg.cholesky(B.todense()))
#         # Prepare to solve linear equation.
#         tol = 1.0e-4
#         maxit = None
#
#         linop = lambda v: (alpha * L * v + AT(A(v)))
#         linop = splin.LinearOperator((n, n), linop)
#
#         [s, info_i] = splin.cg(A=linop, b=-g, x0=None, tol=tol, maxiter=maxit, xtype=None, M=np.dot(R.transpose(), R))
#         if verbose:
#             print('iteration {0:4d}: relative change = {1:.3e}, gradient norm = {2:.3e}'
#                   .format(ii, np.linalg.norm(s[0]) / np.linalg.norm(u), np.linalg.norm(g)))
#             if info_i > 0:
#                 print("WARNING - convergence to tolerance not achieved!")
#             elif info_i < 0:
#                 print("WARNING - illegal input or breakdown")
#
#         # Update current solution
#         u = u + s
#
#     return u / timestep
