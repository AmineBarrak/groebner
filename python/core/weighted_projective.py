"""
python/core/weighted_projective.py
Foundational structures. No changes from validated version.
"""
from sympy import symbols, Poly, Symbol
from math import gcd as int_gcd
from functools import reduce
import warnings


def is_well_formed(weights):
    n = len(weights)
    for i in range(n):
        others = weights[:i] + weights[i+1:]
        if reduce(int_gcd, others) != 1:
            return False
    return True


def weighted_degree(monomial, weights):
    assert len(monomial) == len(weights)
    return sum(w * d for w, d in zip(weights, monomial))


def is_weighted_homogeneous(poly, variables, weights):
    p = Poly(poly, *variables)
    monoms = p.monoms()
    if not monoms:
        return True, 0
    degrees = {weighted_degree(m, weights) for m in monoms}
    if len(degrees) == 1:
        return True, degrees.pop()
    return False, None


def weighted_gcd_int(coords, weights):
    from sympy import factorint
    non_zero = [(abs(c), q) for c, q in zip(coords, weights) if c != 0]
    if not non_zero:
        return 1
    factorizations = [factorint(val) for val, _ in non_zero]
    all_primes = set()
    for f in factorizations:
        all_primes.update(f.keys())
    result = 1
    for p in all_primes:
        min_k = None
        for (val, q), f in zip(non_zero, factorizations):
            vp = f.get(p, 0)
            k = vp // q
            if min_k is None or k < min_k:
                min_k = k
        if min_k and min_k > 0:
            result *= p ** min_k
    return result


class Parametrization:
    def __init__(self, generators, param_vars, coord_vars, weights):
        assert len(generators) == len(coord_vars) == len(weights)
        self.generators  = generators
        self.param_vars  = param_vars
        self.coord_vars  = coord_vars
        self.weights     = weights
        self.n = len(coord_vars) - 1
        self.m = len(param_vars)
        if not is_well_formed(weights):
            warnings.warn(f"Weight vector {weights} is not well-formed.")

    def __repr__(self):
        lines = [f"Parametrization in P({','.join(map(str,self.weights))})"]
        for xi, gi in zip(self.coord_vars, self.generators):
            lines.append(f"  {xi} = {gi}")
        return "\n".join(lines)
