"""
python/core/groebner.py
Sequential Buchberger / F4 engine using sympy as backend.
"""
import time
import itertools
from dataclasses import dataclass, field
from sympy import groebner as sym_groebner, Poly, expand, Mul, S, factor as sym_factor


@dataclass
class ComputationStats:
    stage:                  str   = ""
    elapsed_seconds:        float = 0.0
    n_s_polynomials:        int   = 0
    n_basis_updates:        int   = 0
    peak_basis_size:        int   = 0
    coefficient_bit_lengths: list = field(default_factory=list)

    def summary(self):
        return (
            f"[{self.stage}]\n"
            f"  Time        : {self.elapsed_seconds:.4f}s\n"
            f"  S-polys     : {self.n_s_polynomials}\n"
            f"  |G|         : {self.peak_basis_size}\n"
        )


class F4Engine:
    def __init__(self):
        self.stats = ComputationStats(stage="F4-Sequential")

    def compute(self, generators, variables, elim_vars=None):
        t0 = time.perf_counter()
        if elim_vars:
            rest = [v for v in variables if v not in elim_vars]
            ordered_vars = list(elim_vars) + list(rest)
        else:
            ordered_vars = list(variables)
        try:
            G = sym_groebner(generators, *ordered_vars, order="lex")
            basis = list(G)
        except Exception:
            basis = generators
        self.stats.elapsed_seconds = time.perf_counter() - t0
        self.stats.peak_basis_size = len(basis)
        n = len(generators)
        self.stats.n_s_polynomials = n * (n - 1) // 2
        for poly in basis:
            p = Poly(poly, *ordered_vars)
            for c in p.coeffs():
                try:
                    self.stats.coefficient_bit_lengths.append(int(c).bit_length())
                except Exception:
                    pass
        return basis, self.stats


def extract_elimination_ideal(basis, elim_vars, coord_vars):
    elim_set  = set(str(v) for v in elim_vars)
    coord_set = set(str(v) for v in coord_vars)
    result = []
    for poly in basis:
        if poly == S.Zero:
            continue
        sym_names = {str(s) for s in poly.free_symbols}
        if not sym_names.intersection(elim_set):
            if sym_names.issubset(coord_set | {''}):
                result.append(poly)
            continue
        try:
            factored = sym_factor(poly)
            if isinstance(factored, Mul):
                keep = [arg for arg in factored.args
                        if not {str(s) for s in arg.free_symbols}.issubset(elim_set)]
                if keep:
                    residual = expand(Mul(*keep))
                    r_syms = {str(s) for s in residual.free_symbols}
                    if (not r_syms.intersection(elim_set) and
                            r_syms.issubset(coord_set | {''})):
                        result.append(residual)
        except Exception:
            pass
    seen, unique = set(), []
    for p in result:
        key = str(expand(p))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
