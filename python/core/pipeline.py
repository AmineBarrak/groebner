"""
python/core/pipeline.py
Sequential and parallel three-stage pipeline.
"""
import time
from dataclasses import dataclass, field
from sympy import Poly, expand

from core.weighted_projective import Parametrization, is_weighted_homogeneous
from core.homogenization import homogenize_ideal, parallel_homogenize_ideal
from core.groebner import F4Engine, extract_elimination_ideal
from core.weighted_gcd import normalize_basis, NormalizationStats


# ── WeightedVariety helper (keep here to avoid circular imports) ────────────

class WeightedVariety:
    def __init__(self, ideal_generators, coord_vars, weights):
        self.ideal_generators = ideal_generators
        self.coord_vars       = coord_vars
        self.weights          = weights
        self.degrees          = []
        for f in ideal_generators:
            ok, deg = is_weighted_homogeneous(f, coord_vars, weights)
            if not ok:
                raise ValueError(f"Generator {f} is NOT weighted homogeneous "
                                 f"w.r.t. {weights}")
            self.degrees.append(deg)


@dataclass
class PipelineResult:
    parametrization:      object
    homogenization_result: object  = None
    raw_basis:            list     = field(default_factory=list)
    elimination_basis:    list     = field(default_factory=list)
    normalized_basis:     list     = field(default_factory=list)
    variety:              object   = None
    stage_times:          dict     = field(default_factory=dict)
    total_time:           float    = 0.0
    n_processors:         int      = 1
    is_correct:           object   = None

    def as_dict(self):
        return {
            "n_vars":        self.parametrization.n + 1,
            "n_params":      self.parametrization.m,
            "weights":       str(self.parametrization.weights),
            "n_processors":  self.n_processors,
            "total_time":    self.total_time,
            "t_hom":         self.stage_times.get("homogenization", 0),
            "t_elim":        self.stage_times.get("elimination", 0),
            "t_norm":        self.stage_times.get("normalization", 0),
            "basis_size":    len(self.normalized_basis),
            "is_correct":    self.is_correct,
        }


class WeightedGroebnerPipeline:
    def __init__(self, n_processors=1, use_parallel=False, verbose=True):
        self.n_processors = n_processors
        self.use_parallel = use_parallel and n_processors > 1
        self.verbose      = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def run(self, param):
        t_total = time.perf_counter()
        result  = PipelineResult(parametrization=param, n_processors=self.n_processors)

        # Stage 1
        self._log(f"\n[Stage 1] Homogenization  (p={self.n_processors})")
        t0 = time.perf_counter()
        hom = (parallel_homogenize_ideal(param, self.n_processors)
               if self.use_parallel else homogenize_ideal(param))
        result.stage_times["homogenization"] = time.perf_counter() - t0
        result.homogenization_result = hom
        self._log(f"  {result.stage_times['homogenization']:.4f}s — "
                  f"{len(hom.homogenized_generators)} generators")

        # Stage 2
        self._log(f"\n[Stage 2] F4 Elimination  (p={self.n_processors})")
        t0 = time.perf_counter()
        elim_vars = [hom.alpha_var] + list(param.param_vars)
        engine = F4Engine()
        raw_basis, _ = engine.compute(hom.homogenized_generators,
                                       hom.all_variables,
                                       elim_vars=elim_vars)
        result.stage_times["elimination"] = time.perf_counter() - t0
        result.raw_basis = raw_basis
        elim_basis = extract_elimination_ideal(raw_basis, elim_vars,
                                               list(param.coord_vars))
        result.elimination_basis = elim_basis
        self._log(f"  {result.stage_times['elimination']:.4f}s — "
                  f"{len(elim_basis)} generators")

        # Stage 3
        self._log(f"\n[Stage 3] Normalization  (p={self.n_processors})")
        t0 = time.perf_counter()
        if elim_basis:
            norm_basis, _ = normalize_basis(elim_basis, list(param.coord_vars),
                                            param.weights)
        else:
            norm_basis = []
        result.stage_times["normalization"] = time.perf_counter() - t0
        result.normalized_basis = norm_basis
        self._log(f"  {result.stage_times['normalization']:.4f}s")

        result.total_time = time.perf_counter() - t_total

        if norm_basis:
            try:
                result.variety    = WeightedVariety(norm_basis,
                                                    list(param.coord_vars),
                                                    param.weights)
                result.is_correct = True
            except ValueError as e:
                self._log(f"  WARNING: {e}")
                result.is_correct = False

        self._log(f"\nTotal: {result.total_time:.4f}s")
        return result


class SequentialPipeline(WeightedGroebnerPipeline):
    def __init__(self, verbose=True):
        super().__init__(n_processors=1, use_parallel=False, verbose=verbose)
