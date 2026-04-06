"""
python/core/homogenization.py
Sequential and parallel homogenization (Stage 1).
"""
import time
from dataclasses import dataclass
from sympy import Symbol, expand, cancel, Poly, fraction

from core.weighted_projective import Parametrization


@dataclass
class HomogenizationResult:
    original_generators:    list
    homogenized_generators: list
    alpha_var:              object
    all_variables:          list
    elapsed_seconds:        float = 0.0


def homogenize_generator(xi, gi_expr, param_vars, alpha, weight_qi):
    subs = {t: t / alpha for t in param_vars}
    gi_hom = cancel(gi_expr.subs(subs))
    full_num, full_den = fraction(gi_hom)
    try:
        alpha_den_deg = Poly(full_den, alpha).degree()
    except Exception:
        alpha_den_deg = 0
    clear_factor = alpha ** alpha_den_deg
    rhs_clear = expand(full_num * clear_factor / full_den)
    return expand(clear_factor * xi - rhs_clear)


def homogenize_ideal(param):
    t0 = time.perf_counter()
    alpha = Symbol("alpha", positive=True)
    homogenized, original = [], []
    for xi, gi, qi in zip(param.coord_vars, param.generators, param.weights):
        original.append(xi - gi)
        homogenized.append(homogenize_generator(xi, gi, param.param_vars, alpha, qi))
    all_vars = [alpha] + list(param.param_vars) + list(param.coord_vars)
    return HomogenizationResult(original, homogenized, alpha,
                                all_vars, time.perf_counter() - t0)


# ── Parallel version ────────────────────────────────────────────────────────

def _homogenize_chunk(args):
    chunk_indices, coord_names, gen_strs, param_names, weights, alpha_name = args
    from sympy import Symbol, sympify
    alpha = Symbol(alpha_name, positive=True)
    param_vars = [Symbol(n) for n in param_names]
    coord_vars = [Symbol(n) for n in coord_names]
    local = {n: Symbol(n) for n in param_names + coord_names + [alpha_name]}
    results = []
    for idx, i in enumerate(chunk_indices):
        xi = coord_vars[i]
        gi = sympify(gen_strs[i], locals=local)
        fi_hom = homogenize_generator(xi, gi, param_vars, alpha, weights[idx])
        results.append((i, fi_hom))
    return results


def parallel_homogenize_ideal(param, n_processors=4):
    import multiprocessing as mp
    t0 = time.perf_counter()
    n = len(param.generators)
    if n <= 6 or n_processors <= 1:
        r = homogenize_ideal(param)
        r.elapsed_seconds = time.perf_counter() - t0
        return r
    indices = list(range(n))
    chunk_size = max(1, n // n_processors)
    chunks = [indices[i:i+chunk_size] for i in range(0, n, chunk_size)]
    coord_names  = [str(v) for v in param.coord_vars]
    param_names  = [str(v) for v in param.param_vars]
    gen_strs     = [str(g) for g in param.generators]
    args_list = [
        (chunk, coord_names, gen_strs,
         param_names, [param.weights[i] for i in chunk], "alpha")
        for chunk in chunks
    ]
    with mp.Pool(processes=min(n_processors, len(chunks))) as pool:
        nested = pool.map(_homogenize_chunk, args_list)
    all_res = sorted([item for sub in nested for item in sub], key=lambda x: x[0])
    homogenized = [r[1] for r in all_res]
    original    = [x - g for x, g in zip(param.coord_vars, param.generators)]
    alpha = Symbol("alpha", positive=True)
    all_vars = [alpha] + list(param.param_vars) + list(param.coord_vars)
    return HomogenizationResult(original, homogenized, alpha, all_vars,
                                time.perf_counter() - t0)
