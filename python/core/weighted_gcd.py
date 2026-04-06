"""
python/core/weighted_gcd.py
Weighted GCD computation and coefficient normalization (Stage 3).
"""
import time
from dataclasses import dataclass, field
from math import gcd as int_gcd
from sympy import Poly, Rational, S

from core.weighted_projective import weighted_gcd_int


@dataclass
class NormalizationStats:
    n_polynomials:        int   = 0
    n_normalized:         int   = 0
    max_coeff_bits_before: int  = 0
    max_coeff_bits_after:  int  = 0
    elapsed_seconds:      float = 0.0
    wgcd_values:          list  = field(default_factory=list)

    def summary(self):
        return (
            f"[Normalization]\n"
            f"  Polys processed : {self.n_polynomials}\n"
            f"  Normalized      : {self.n_normalized}\n"
            f"  Bits before     : {self.max_coeff_bits_before}\n"
            f"  Bits after      : {self.max_coeff_bits_after}\n"
            f"  Time            : {self.elapsed_seconds:.4f}s\n"
        )


def normalize_basis(basis, variables, weights, parallel=False, n_processors=4):
    """Normalize each polynomial in the basis by dividing out the content
    (ordinary GCD of its integer coefficients).  For a weighted homogeneous
    polynomial, dividing by a constant preserves homogeneity, so this is the
    correct normalization: content removal first, then optionally apply the
    full weighted GCD across the *coordinate evaluations* of the variety
    (which is a separate, point-level operation).

    The previous version incorrectly passed the ambient-space weight vector
    to weighted_gcd_int, slicing or padding it to match the coefficient count.
    Since the weights correspond to coordinate variables, not to monomial
    positions in a single polynomial, this produced wrong pairings.

    The correct per-polynomial normalization is simply: divide all
    coefficients by their ordinary GCD (the content of the polynomial).
    """
    t0 = time.perf_counter()
    stats = NormalizationStats(n_polynomials=len(basis))
    normalized = []

    for poly_expr in basis:
        p = Poly(poly_expr, *variables)
        coeffs = p.coeffs()
        int_coeffs = []
        for c in coeffs:
            try:
                int_coeffs.append(int(c))
            except Exception:
                int_coeffs = []
                break
        if not int_coeffs:
            normalized.append(poly_expr)
            continue
        for c in int_coeffs:
            bl = abs(c).bit_length() if abs(c) > 0 else 0
            if bl > stats.max_coeff_bits_before:
                stats.max_coeff_bits_before = bl

        # Compute the content: ordinary GCD of all integer coefficients.
        # This is the correct normalization for a single polynomial.
        from functools import reduce
        d = reduce(int_gcd, [abs(c) for c in int_coeffs if c != 0], 0)

        stats.wgcd_values.append(d)
        if d > 1:
            stats.n_normalized += 1
            new_coeffs = {m: Rational(int(c), d)
                          for m, c in zip(p.monoms(), coeffs)}
            norm_poly = Poly(new_coeffs, *variables, domain='QQ').as_expr()
        else:
            norm_poly = poly_expr
        normalized.append(norm_poly)
        for c in Poly(norm_poly, *variables).coeffs():
            try:
                bl = abs(int(c)).bit_length()
                if bl > stats.max_coeff_bits_after:
                    stats.max_coeff_bits_after = bl
            except Exception:
                pass

    stats.elapsed_seconds = time.perf_counter() - t0
    return normalized, stats
