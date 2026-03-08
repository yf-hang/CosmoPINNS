#import torch
#import numpy as np
import mpmath as mp
#from plot_tools.plot_bc_points import visualize_bc_points #visualize_collocation_points

# ---------------- letters ----------------
def _ws(x1: float, x2: float, c: float):
    x1m, x2m, cm = mp.mpf(x1), mp.mpf(x2), mp.mpf(c)
    w1 = x1m + cm
    w2 = x2m + cm
    w3 = x1m - cm
    w4 = x2m - cm
    w5 = x1m + x2m
    return w1, w2, w3, w4, w5

def _w_select(x1: float, x2: float, c: float, *idx):
    ws = _ws(x1, x2, c)
    return tuple(ws[i - 1] for i in idx)

def _eps_to_n_int(eps: float, tol=1e-12):
    epsm = mp.mpf(eps)
    if mp.fabs(epsm + 1) < tol: return 1
    if mp.fabs(epsm + 2) < tol: return 2
    if mp.fabs(epsm + 3) < tol: return 3
    return None

def _eps_to_n_half(eps: float, tol=1e-12):
    epsm = mp.mpf(eps)
    if mp.fabs(epsm + mp.mpf("0.5")) < tol: return 1
    if mp.fabs(epsm + mp.mpf("1.5")) < tol: return 2
    if mp.fabs(epsm + mp.mpf("2.5")) < tol: return 3
    if mp.fabs(epsm + mp.mpf("3.5")) < tol: return 4
    return None

def eps_to_n_int(eps):
    return _eps_to_n_int(eps)

def eps_to_n_half(eps):
    return _eps_to_n_half(eps)

# ---------------- coefficients ----------------
def alpha0(eps: float):
    return (mp.pi ** 2) / (mp.sin(mp.pi * eps) ** 2)

def alpha1(eps: float):
    return - (mp.pi ** 2) / (mp.sin(mp.pi * eps) * mp.sin(2.0 * mp.pi * eps))

def alpha2(eps: float):
    return (2.0 ** (-2.0 * eps - 1.0)
            * mp.sqrt(mp.pi)
            * (1.0 / mp.sin(mp.pi * eps))
            * mp.gamma(eps)
            * mp.gamma(0.5 - eps))

# ------------------------------------------------------
# ---------------- general eps solutions ---------------
# ------------------------------------------------------
def I1_eps(x1: float, x2: float, eps: float, c: float, *, _alp0=None, _alp2=None):
    w1, w2, w3, w4, w5 = _w_select(x1, x2, c, 1, 2, 3, 4, 5)

    if _alp0 is None:
        _alp0 = alpha0(eps)
    if _alp2 is None:
        _alp2 = alpha2(eps)

    term1 = _alp0 * (w1 ** eps) * (w2 ** eps)

    # 2F1(1, -2eps; 1-eps; wi / wj)
    def _hg(z):
        return mp.hyper([1.0, -2.0 * eps], [1.0 - eps], z)

    term2 = _alp2 * (w5 ** (2.0 * eps)) * (_hg(w3 / w5) + _hg(w4 / w5))

    return term1 - term2

def I2_eps(x1: float, x2: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    w2, w3, w5 = _w_select(x1, x2, c, 2, 3, 5)

    if _alp1 is None:
        _alp1 = alpha1(eps)
    if _alp2 is None:
        _alp2 = alpha2(eps)

    term1 = _alp1 * (w2 ** eps) * (w3 ** eps)

    u1 = w3 / w2
    u2 = w5 / w2
    hg = mp.hyper([eps, 2.0 * eps], [1.0 + 2.0 * eps], u2)
    term2 = _alp2 * (u1 ** eps) * (w5 ** (2.0 * eps)) * hg

    return term1 + term2

def I3_eps(x1: float, x2: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    w1, w4, w5 = _w_select(x1, x2, c, 1, 4, 5)

    if _alp1 is None:
        _alp1 = alpha1(eps)
    if _alp2 is None:
        _alp2 = alpha2(eps)

    term1 = _alp1 * (w1 ** eps) * (w4 ** eps)

    u1 = w4 / w1
    u2 = w5 / w1
    hg = mp.hyper([eps, 2.0 * eps], [1.0 + 2.0 * eps], u2)
    term2 = _alp2 * (u1 ** eps) * (w5 ** (2.0 * eps)) * hg

    return term1 + term2

def I4_eps(x1: float, x2: float, eps: float, c: float, *, _alp2=None):
    (w5,) = _w_select(x1, x2, c, 5)

    if _alp2 is None:
        _alp2 = alpha2(eps)

    return 2.0 * _alp2 * (w5 ** (2.0 * eps))

# -----------------------------------------
# ---------------- eps = 0 ----------------
# -----------------------------------------
def I1_eps0(x1: float, x2: float, c: float):
    w1, w2, w3, w4, w5 = _w_select(x1, x2, c, 1, 2, 3, 4, 5)

    return (-(mp.pi**2)/6
            + mp.polylog(2, w3 / w1)
            + mp.polylog(2, w4 / w2)
            - mp.polylog(2, (w3 * w4) / (w1 * w2)))

def I2_eps0(x1: float, x2: float, c: float):
    w2, w5 = _w_select(x1, x2, c, 2, 5)

    return mp.log(w5 / w2)

def I3_eps0(x1: float, x2: float, c: float):
    w1, w5 = _w_select(x1, x2, c, 1, 5)

    return mp.log(w5 / w1)

def I4_eps0(_x1: float, _x2: float, _c: float):
    return mp.mpf("1")

# ------------------------------------------
# ---------------- eps = -n ----------------
# ------------------------------------------
def _jacobi_P(n: int, x):
    z = (1 - x) / 2
    return mp.binomial(2*n - 1, n - 1) * mp.hyper([1 - n, n], [n + 1], z)

def _int_branch_series_sum(w_main, w5, n: int):
    # Sum_{k=0}^{n-1} C(n+k,k) * w_main^k * w5^(n-k-1)
    return mp.fsum(
        mp.binomial(n + k, k) * (w_main ** k) * (w5 ** (n - k - 1))
        for k in range(n)
    )

def I1_eps_int(x1: float, x2: float, n: int, c: float):
    w1, w2, w3, w4, w5 = _w_select(x1, x2, c, 1, 2, 3, 4, 5)

    term1 = 1 / (w1**n * w2**n)

    x1 = 1 - 2 * w3 / w5
    x2 = 1 - 2 * w4 / w5
    term2 =  _jacobi_P(n, x1) / (w2**n * w5**n) + _jacobi_P(n, x2) / (w1**n * w5**n)

    return term1 - term2

def I2_eps_int(x1: float, x2: float, n: int, c: float):
    w2, w3, w5 = _w_select(x1, x2, c, 2, 3, 5)

    sum_w2w5 = _int_branch_series_sum(w2, w5, n)

    return -(w3 / (w2**n * w5**(2*n))) * sum_w2w5

def I3_eps_int(x1: float, x2: float, n: int, c: float):
    w1, w4, w5 = _w_select(x1, x2, c, 1, 4, 5)

    sum_w1w5 = _int_branch_series_sum(w1, w5, n)

    return -(w4 / (w1**n * w5**(2*n))) * sum_w1w5

def I4_eps_int(x1: float, x2: float, n: int, c: float):
    (w5,) = _w_select(x1, x2, c, 5)

    return mp.binomial(2*n, n) / (w5**(2*n))


# ---------------- eps = -(2n-1)/2 ----------------
def _odd_double_factorial(m: int):
    # m must be odd positive
    out = mp.mpf("1")
    for k in range(1, m + 1, 2):
        out *= k
    return out

def _Poly_half(n: int, a, b):
    a = mp.mpf(a) if isinstance(a, (int, float)) else a
    b = mp.mpf(b) if isinstance(b, (int, float)) else b

    if n == 1:
        return mp.mpf("1")
    if n == 2:
        return 3*(a**2) + 8*a*b - 3*(b**2)
    if n == 3:
        return 15*(a**4) + 70*(a**3)*b + 128*(a**2)*(b**2) - 70*a*(b**3) - 15*(b**4)
    if n == 4:
        return (105*(a**6) + 700*(a**5)*b + 1981*(a**4)*(b**2) + 3072*(a**3)*(b**3)
                - 1981*(a**2)*(b**4) - 700*a*(b**5) - 105*(b**6))
    raise ValueError("Half-integer branch supports n=(1,2,3,4) only.")

def _eps_half_core(a, b, w5, n: int):
    df = _odd_double_factorial(2 * n - 1)
    pow_ab = a ** (n - 0.5) * b ** (n - 0.5)
    term1 = (1 / pow_ab) * mp.atan(mp.sqrt(b / a))

    poly = _Poly_half(n, a, b)
    term2 = poly / (df * (pow_ab * w5 ** (2 * n - 1)))

    return -2 * mp.pi * (term1 - term2)

def I2_eps_half(x1: float, x2: float, n: int, c: float):
    w2, w3, w5 = _w_select(x1, x2, c, 2, 3, 5)

    return _eps_half_core(w2, w3, w5, n)

def I3_eps_half(x1: float, x2: float, n: int, c: float):
    w1, w4, w5 = _w_select(x1, x2, c, 1, 4, 5)

    return _eps_half_core(w1, w4, w5, n)

def I4_eps_half(x1: float, x2: float, n: int, c: float):
    (w5,) = _w_select(x1, x2, c, 5)
    coeff = (2**(4*n - 1)) * mp.pi / (n * mp.binomial(2*n, n))

    return coeff / (w5**(2*n - 1))

def I1_eps_half(x1: float, x2: float, n: int, c: float):
    w1, w2, w3, w4, w5 = _w_select(x1, x2, c, 1, 2, 3, 4, 5)
    I2t = I2_eps_half(x1, x2, n, c)
    I3t = I3_eps_half(x1, x2, n, c)
    I4t = I4_eps_half(x1, x2, n, c)

    return (mp.pi**2) / (w1 ** (n - 0.5) * w2 ** (n - 0.5)) + I2t + I3t - I4t

# ---------------- final solutions ----------------
EPS_TOL = mp.mpf("1e-12")

def I1_fin(x1: float, x2: float, eps: float, c: float, *, _alp0=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I1_eps0(x1, x2, c)

    n_int = _eps_to_n_int(eps)
    if n_int is not None:
        return I1_eps_int(x1, x2, n_int, c)

    n_half = _eps_to_n_half(eps)
    if n_half is not None:
        return I1_eps_half(x1, x2, n_half, c)

    return I1_eps(x1, x2, eps, c, _alp0=_alp0, _alp2=_alp2)


def I2_fin(x1: float, x2: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I2_eps0(x1, x2, c)

    n_int = _eps_to_n_int(eps)
    if n_int is not None:
        return I2_eps_int(x1, x2, n_int, c)

    n_half = _eps_to_n_half(eps)
    if n_half is not None:
        return I2_eps_half(x1, x2, n_half, c)

    return I2_eps(x1, x2, eps, c, _alp1=_alp1, _alp2=_alp2)


def I3_fin(x1: float, x2: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I3_eps0(x1, x2, c)

    n_int = _eps_to_n_int(eps)
    if n_int is not None:
        return I3_eps_int(x1, x2, n_int, c)

    n_half = _eps_to_n_half(eps)
    if n_half is not None:
        return I3_eps_half(x1, x2, n_half, c)

    return I3_eps(x1, x2, eps, c, _alp1=_alp1, _alp2=_alp2)


def I4_fin(x1: float, x2: float, eps: float, c: float, *, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I4_eps0(x1, x2, c)

    n_int = _eps_to_n_int(eps)
    if n_int is not None:
        return I4_eps_int(x1, x2, n_int, c)

    n_half = _eps_to_n_half(eps)
    if n_half is not None:
        return I4_eps_half(x1, x2, n_half, c)

    return I4_eps(x1, x2, eps, c, _alp2=_alp2)