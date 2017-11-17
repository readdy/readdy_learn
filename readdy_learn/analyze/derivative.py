import numpy as np

import scipy.integrate as integrate
from scipy import sparse
from scipy.sparse import linalg as splin
import scipy.optimize as so
from pynumtools.lgmres import lgmres
import collections
import matplotlib.pyplot as plt

from sklearn.decomposition import FactorAnalysis


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
        if chopright and ix == len(xs) - 1:
            continue
        x = xs[ix]
        coeff = fd_coeff(x, wx, k=1)
        D_data[offset:offset + len(coeff)] = coeff
        D_row_data[offset:offset + len(coeff)] = np.ones(shape=(len(coeff),)) * ix
        D_col_data[offset:offset + len(coeff)] = np.arange(max((0, ix - wwidth)), min(ix + wwidth + 1, n_nodes))
        offset += len(coeff)

    if chopright:
        return sparse.bsr_matrix((D_data, (D_row_data, D_col_data)), shape=(n_nodes - 1, n_nodes))
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
    return sparse.csc_matrix((D_data, (D_row_data, D_col_data)), shape=(n_nodes - 1, n_nodes))


def get_integration_operator(xs):
    n_nodes = len(xs)
    D_data = np.zeros(shape=(int(.5 * (n_nodes ** 2 + n_nodes) - 1),))
    D_row_data = np.empty_like(D_data)
    D_col_data = np.empty_like(D_data)

    current_row = np.array([0], dtype=np.float64)

    offset = 0
    for ix in range(n_nodes):
        # ix'th row
        if ix == 0:
            # ignore 1st row, its const zero
            pass
        else:
            # fill up data with current row...
            d = xs[ix] - xs[ix - 1]
            current_row[-1] += d
            current_row = np.append(current_row, d)
            D_data[offset:offset + len(current_row)] = current_row
            D_row_data[offset:offset + len(current_row)] = ix
            D_col_data[offset:offset + len(current_row)] = np.array([range(len(current_row))])
            offset += len(current_row)

    D_data = .5 * D_data

    assert (len(D_data) == offset)

    return sparse.csc_matrix((D_data, (D_row_data, D_col_data)), shape=(n_nodes, n_nodes))


def get_integration_adjoint_operator(xs):
    """
    This guy integrates from x to L, so the first row contains the full thing,
    second row full thing minus the last node etc
    """
    n_nodes = len(xs)
    D_data = np.zeros(shape=(int(.5 * (n_nodes ** 2 + n_nodes) - 1),))
    D_row_data = np.empty_like(D_data)
    D_col_data = np.empty_like(D_data)

    current_row = np.array([0], dtype=np.float64)

    offset = 0
    for ix in range(n_nodes - 1, -1, -1):
        # ix'th row
        if ix == n_nodes - 1:
            # ignore last row, its const zero
            pass
        else:
            # fill up data with current row...
            d = xs[ix] - xs[ix - 1]
            current_row[-1] += d
            current_row = np.append(current_row, d)
            D_data[offset:offset + len(current_row)] = current_row
            D_row_data[offset:offset + len(current_row)] = ix
            D_col_data[offset:offset + len(current_row)] = np.array(
                [range(n_nodes - 1, n_nodes - len(current_row) - 1, -1)])
            offset += len(current_row)

    D_data = .5 * D_data
    assert (len(D_data) == offset)
    return sparse.csc_matrix((D_data, (D_row_data, D_col_data)), shape=(n_nodes, n_nodes))


def trapz(xs, ys):
    result = np.empty_like(xs)
    result[0] = 0
    for ix in range(1, len(xs)):
        result[ix] = .5 * (xs[ix] - xs[ix - 1]) * (ys[ix - 1] + ys[ix])
    return result


def ld_derivative(data, xs, alpha, maxit=1000, linalg_solver_maxit=100, tol=1e-4, atol=1e-4, rtol=1e-6, verbose=False,
                  show_progress=True, solver='lgmres', precondition=True):
    assert isinstance(data, np.ndarray)

    label = None
    if show_progress:
        from ipywidgets import IntProgress, Label, Box
        from IPython.display import display
        label = Label("Progress: 0/{} it, atol={}/{}, rtol={}/{}".format(0, '?', atol, '?', rtol))
        box = Box([label])
        display(box)

    if show_progress:
        label.value = 'copying data and setting f(0) = 0'

    # require f(0) = 0
    data = np.copy(data) - data[0]

    data = data.squeeze()
    assert len(data.shape) == 1

    epsilon = 1e-4

    n = len(data)

    # differentiation operator
    if show_progress:
        label.value = 'obtaining differentiation operator'
    D = get_differentiation_operator_midpoint(xs)
    D_T = D.transpose().tocsc()

    def A(v):
        # integrate v from 0 to x
        return integrate.cumtrapz(y=v, x=xs, initial=0)

    def A_adjoint(w):
        # integrate w from x to L <=> int_0^Lw - int_0^xw
        full_integral = integrate.trapz(w, xs)
        # assert np.isclose(full_integral, A(w)[-1]), "got {}, but expected {}".format(full_integral, A(w)[-1])
        return np.ones_like(w) * full_integral - integrate.cumtrapz(w, xs, initial=0)

    Aadj_A = lambda v: A_adjoint(A(v))

    if show_progress:
        label.value = 'calculating native gradient of data'
    u = np.gradient(data, xs)

    if show_progress:
        label.value = 'precalculating A*(data)'
    KT_data = A_adjoint(data)

    xs_diff = np.diff(xs)
    # DX = sparse.spdiags(np.diff(xs), 0, n - 1, n - 1)

    if show_progress:
        label.value = 'preparing E_n'
    E_n = sparse.dia_matrix((n - 1, n - 1), dtype=xs.dtype)
    prev_grad_norm = None

    spsolve_term = None
    if precondition or solver == 'spsolve' or solver == 'np' or True:
        if show_progress:
            if precondition:
                label.value = 'computing preconditioner: getting integration operator'
            else:
                label.value = 'assembling matrix'
        K = get_integration_operator(xs)
        if show_progress and precondition:
            label.value = 'computing preconditioner: getting adjoint integration operator'
        KT = get_integration_adjoint_operator(xs)
        if show_progress and precondition:
            label.value = 'calculating K^T * K'
        spsolve_term = KT * K

    if show_progress:
        label.value = 'begin solver loop'
    # Main loop.
    relative_change = None
    first_strike = False
    for ii in range(1, maxit + 1):
        # Diagonal matrix of weights, for linearizing E-L equation.
        E_n.setdiag(xs_diff * (1. / np.sqrt(np.diff(u) ** 2.0 + epsilon)))
        # Q = E_n #DX *
        L = D_T * E_n * D
        # Gradient of functional.
        g = Aadj_A(u) - KT_data
        g = g + alpha * L * u

        # solve linear equation.
        info_i = 0
        if solver == 'lgmres' or solver == 'lgmres_scipy':
            linop = splin.LinearOperator((n, n), lambda v: (alpha * L * v + Aadj_A(v)))
            if precondition:
                if show_progress:
                    label.value = 'Progress: {}/{} it, atol={}/{}, rtol={}/{}, compute spilu' \
                        .format(ii, maxit, prev_grad_norm, atol, relative_change, rtol)
                lu = splin.spilu(alpha * L + spsolve_term, drop_tol=5e-2)
                precond = splin.LinearOperator((n, n), lambda v: lu.solve(v))
                if show_progress:
                    label.value = 'Progress: {}/{} it, atol={}/{}, rtol={}/{}, lgmres' \
                        .format(ii, maxit, prev_grad_norm, atol, relative_change, rtol)
                if solver == 'lgmres_scipy':
                    s, info_i = splin.lgmres(A=linop, b=-g, x0=u, tol=tol, maxiter=linalg_solver_maxit, M=precond)
                else:
                    s = lgmres(A=linop, b=-g, x0=u, tol=tol, maxiter=linalg_solver_maxit, M=precond)
            else:
                if solver == 'lgmres_scipy':
                    s, info_i = splin.lgmres(A=linop, b=-g, x0=u, tol=tol, maxiter=linalg_solver_maxit, outer_k=7)
                else:
                    s = lgmres(A=linop, b=-g, x0=u, tol=tol, maxiter=linalg_solver_maxit)
        elif solver == 'bicgstab':
            linop = splin.LinearOperator((n, n), lambda v: (alpha * L * v + Aadj_A(v)))
            if precondition:
                if show_progress:
                    label.value = 'Progress: {}/{} it, atol={}/{}, rtol={}/{}, compute spilu' \
                        .format(ii, maxit, prev_grad_norm, atol, relative_change, rtol)
                lu = splin.spilu(alpha * L + spsolve_term, drop_tol=5e-2)
                precond = splin.LinearOperator((n, n), lambda v: lu.solve(v))
                if show_progress:
                    label.value = 'Progress: {}/{} it, atol={}/{}, rtol={}/{}, bicgstab' \
                        .format(ii, maxit, prev_grad_norm, atol, relative_change, rtol)
                [s, info_i] = splin.bicgstab(A=linop, b=-g, x0=u, tol=tol, maxiter=linalg_solver_maxit, M=precond)
            else:
                [s, info_i] = splin.bicgstab(A=linop, b=-g, x0=u, tol=tol, maxiter=linalg_solver_maxit)
        elif solver == 'spsolve':
            # todo this behaves oddly
            s = splin.spsolve((alpha * L + spsolve_term), -g, use_umfpack=False, permc_spec='COLAMD')
        elif solver == 'np':
            # todo this behaves oddly
            s = np.linalg.solve((alpha * L + spsolve_term).todense().astype(np.float64), (-g).astype(np.float64))

        relative_change = np.linalg.norm(s[0]) / np.linalg.norm(u)
        if verbose:
            print('iteration {0:4d}: relative change = {1:.3e}, gradient norm = {2:.3e}'
                  .format(ii, relative_change, np.linalg.norm(g)))
            if info_i > 0:
                print("WARNING - convergence to tolerance not achieved!")
            elif info_i < 0:
                print("WARNING - illegal input or breakdown")

        #if prev_grad_norm is not None and np.linalg.norm(g) > prev_grad_norm:
        #    linalg_solver_maxit = int(2 * linalg_solver_maxit)
        #else:
        #    linalg_solver_maxit = int(.95 * linalg_solver_maxit)

        if prev_grad_norm is not None and np.linalg.norm(g) > prev_grad_norm and np.linalg.norm(g) > 1:
            print("WARNING - increasing large gradient norm: {} -> {}".format(prev_grad_norm, np.linalg.norm(g)))
            if not first_strike:
                break
            first_strike = True
        else:
            first_strike = False

        prev_grad_norm = np.linalg.norm(g)

        # Update current solution
        u = u + s

        if show_progress:
            label.value = "Progress: {}/{} it, atol={}/{}, rtol={}/{}" \
                .format(ii, maxit, np.linalg.norm(g), atol, relative_change, rtol)

        if atol is not None and np.linalg.norm(g) < atol:
            label.value = "ld derivative reached atol = {} < {}, finish".format(atol, np.linalg.norm(g))
            break
        if rtol is not None and relative_change < rtol:
            label.value = "ld derivative reached rtol = {} < {}, finish".format(rtol, relative_change)
            break

    if show_progress:
        box.close()

    return u


def test_finite_differences():
    x0 = np.arange(0, 2.0 * np.pi, 0.05)
    print(len(x0))
    xx = []
    for x in x0:
        if np.random.random() < .8:
            xx.append(x)
    x0 = np.array(xx)
    x0 = np.arange(0, 2.0 * np.pi, 0.05)

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


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def mse(x, y):
    assert x.shape == y.shape
    return ((x - y) ** 2).mean(axis=None)


def estimate_alpha(f):
    # calculate MSE between Au* and f, should be equal to variance of noise in f
    fa = FactorAnalysis()
    fa.fit(f.reshape(-1, 1))
    return fa.noise_variance_


def estimate_noise_variance(xs, ys):
    fun = lambda t, b, c, d, e, g: b * np.exp(c * t) + d * t + e * t * t + g * t * t * t
    dfun_db = lambda t, b, c, d, e, g: np.exp(c * t)
    dfun_dc = lambda t, b, c, d, e, g: t * b * np.exp(c * t)
    dfun_dd = lambda t, b, c, d, e, g: t
    dfun_de = lambda t, b, c, d, e, g: t * t
    dfun_dg = lambda t, b, c, d, e, g: t * t * t
    derivatives = [dfun_db, dfun_dc, dfun_dd, dfun_de, dfun_dg]

    def jac(t, b, c, d, e, g):
        result = np.array([np.array(f(t, b, c, d, e, g)) for f in derivatives])
        return result.T

    copt, _ = so.curve_fit(fun, xs, ys, maxfev=300000, jac=jac)

    ff = lambda t: fun(t, *copt)
    return np.var(ff(xs) - ys, ddof=1)


def score_ld_derivative(alpha, xs, ys, **kw):
    if not 'maxit' in kw.keys():
        kw['maxit'] = 1000
    if not 'linalg_solver_maxit' in kw.keys():
        kw['linalg_solver_maxit'] = 10000
    if not 'verbose' in kw.keys():
        kw['verbose'] = False
    if not 'solver' in kw.keys():
        kw['solver'] = 'lgmres'
    if not 'tol' in kw.keys():
        kw['tol'] = 1e-12
    if not 'atol' in kw.keys():
        kw['atol'] = 1e-4
    if not 'rtol' in kw.keys():
        kw['rtol'] = 1e-6
    ld_deriv = ld_derivative(ys, xs, alpha=alpha, **kw)
    integrated_ld = integrate.cumtrapz(ld_deriv, x=xs, initial=ys[0])
    return np.abs(estimate_noise_variance(xs, ys)-mse(integrated_ld, ys))


def test_ld_derivative():
    x0 = np.arange(0, 2.0 * np.pi, 0.005)
    xx = []
    for x in x0:
        if np.random.random() < .4:
            xx.append(x)
    x0 = np.array(xx)
    # x0 = np.arange(0, 2.0 * np.pi, 0.005)

    testf = np.array([np.sin(x) for x in x0])
    testf = testf + np.random.normal(0.0, 0.04, x0.shape)
    print("estimated noise variance: {}".format(estimate_noise_variance(x0, testf)))
    true_deriv = [np.cos(x) for x in x0]

    if True:
        ld_deriv = ld_derivative(testf, x0, alpha=.04 ** 2, maxit=1000, linalg_solver_maxit=10000, verbose=True,
                                 solver='spsolve', precondition=False, tol=1e-12, atol=1e-4, rtol=1e-6)

        plt.plot(x0, testf, label='f')
        plt.plot(x0, true_deriv, label='df')
        # plt.plot(deriv, label='approx df')
        # plt.plot(dmidxs, Dmidderiv)
        # plt.plot(ld_derivative(testf, x0, alpha=5e-4, verbose=True, solver='lgmres'), label='total variation df alpha=5e-4')
        # plt.plot(ld_derivative(testf, x0, alpha=1e-3, verbose=True, solver='lgmres'), label='total variation df alpha=1e-3')
        # plt.plot(ld_derivative(testf, x0, alpha=1e-2, verbose=True, solver='lgmres'),
        #         label='total variation df alpha=1e-2')
        plt.plot(x0, ld_deriv, label='umfpack')
        integrated_ld = integrate.cumtrapz(ld_deriv, x=x0, initial=testf[0])
        plt.plot(x0, integrated_ld, label='integrated')
        print("mse between integrated ld and original fun: {}".format(mse(integrated_ld, testf)))
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
