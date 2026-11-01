"""QC Report structure and quality tier classification."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from gdc.core.coreg import CoregResult


@dataclass
class QCReport:
    """Comprehensive DEM Quality Control Report."""

    tier: str  # "reliable" | "cautious" | "unreliable"
    tier_reasons: List[str]
    coreg_result: CoregResult
    uncertainty: Dict[str, float]
    multi_start_spread: Optional[Dict[str, float]]
    spatial_coverage: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def evaluate(
        cls,
        coreg_result: CoregResult,
        bounds_px: Tuple[Tuple[float, float], Tuple[float, float]],
        points_x: np.ndarray,
        points_y: np.ndarray,
        multi_start_spread: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "QCReport":
        """Assess coregistration outcome and assign QC tier."""
        reasons: List[str] = []
        is_unreliable = False
        is_cautious = False

        stats_after = coreg_result.stats_after
        valid_n = stats_after.get("count", 0)
        total_n = stats_after.get("total_count", valid_n)
        outlier_n = stats_after.get("outlier_count", 0)

        # 1. Sample size checks (PRD §7: < 1000 points may risk local minima)
        if valid_n < 300:
            is_unreliable = True
            reasons.append(f"Insufficient valid points (N={valid_n} < 300); high risk of spurious convergence.")
        elif valid_n < 1000:
            is_cautious = True
            reasons.append(f"Sparse point sample (300 <= N={valid_n} < 1000); results may have sensitivity to local minima.")

        # 2. Outlier ratio checks
        if total_n > 0:
            outlier_ratio = outlier_n / total_n
            if outlier_ratio > 0.25:
                is_unreliable = True
                reasons.append(f"Excessive outlier ratio ({outlier_ratio:.1%} > 25%).")
            elif outlier_ratio > 0.10:
                is_cautious = True
                reasons.append(f"Elevated outlier ratio ({outlier_ratio:.1%} > 10%).")

        # 3. Boundary check (did optimization hit hard parameter bounds?)
        b_x, b_y = bounds_px
        dx = coreg_result.dx_px
        dy = coreg_result.dy_px
        hit_margin = 0.05
        hit_x = (dx <= b_x[0] + hit_margin) or (dx >= b_x[1] - hit_margin)
        hit_y = (dy <= b_y[0] + hit_margin) or (dy >= b_y[1] - hit_margin)
        if hit_x or hit_y:
            is_unreliable = True
            reasons.append(f"Optimization reached boundary limits (dx={dx:.3f} in {b_x}, dy={dy:.3f} in {b_y}). Search range may be too narrow.")

        # 4. Convergence and optimizer success
        if not coreg_result.optimizer_result.success:
            is_cautious = True
            reasons.append("Optimizer indicated non-standard convergence; verify residual distribution.")

        # 5. NMAD divergence check
        nmad_b = coreg_result.nmad_before
        nmad_a = coreg_result.nmad_after
        if np.isfinite(nmad_b) and np.isfinite(nmad_a):
            if nmad_a > nmad_b * 1.05:
                is_unreliable = True
                reasons.append(f"Residual NMAD increased after shift ({nmad_b:.3f} -> {nmad_a:.3f} m).")
        else:
            is_unreliable = True
            reasons.append("Residual NMAD is non-finite.")

        # 6. Multi-start consistency check (if performed)
        if multi_start_spread is not None:
            spread_px = multi_start_spread.get("max_spread_px", 0.0)
            if spread_px > 0.5:
                is_unreliable = True
                reasons.append(f"Multiple starting points converged to disparate minima (spread = {spread_px:.2f} px).")
            elif spread_px > 0.1:
                is_cautious = True
                reasons.append(f"Multiple starting points exhibited minor local variation (spread = {spread_px:.2f} px).")

        # Assign final tier
        if is_unreliable:
            tier = "unreliable"
        elif is_cautious:
            tier = "cautious"
        else:
            tier = "reliable"
            reasons.append("Convergence confirmed: adequate points, clean residual reduction, and within search bounds.")

        # Uncertainty estimation
        nmad_val = max(coreg_result.nmad_after, 0.001)
        sqrt_n = np.sqrt(max(valid_n, 1))
        # Standard error of the median for normal-like error distribution is ~ 1.253 * sigma / sqrt(N)
        se_z = float(1.2533 * nmad_val / sqrt_n)
        se_x = float(nmad_val / sqrt_n)
        se_y = float(nmad_val / sqrt_n)
        uncertainty = {
            "se_dx_m": round(se_x, 4),
            "se_dy_m": round(se_y, 4),
            "se_dz_m": round(se_z, 4),
            "nmad_m": round(nmad_val, 4),
        }

        # Spatial coverage
        valid_pts_mask = np.isfinite(points_x) & np.isfinite(points_y)
        if np.any(valid_pts_mask):
            min_x, max_x = float(np.min(points_x[valid_pts_mask])), float(np.max(points_x[valid_pts_mask]))
            min_y, max_y = float(np.min(points_y[valid_pts_mask])), float(np.max(points_y[valid_pts_mask]))
            bbox = {
                "min_x": round(min_x, 2),
                "max_x": round(max_x, 2),
                "min_y": round(min_y, 2),
                "max_y": round(max_y, 2),
                "span_x_m": round(max_x - min_x, 2),
                "span_y_m": round(max_y - min_y, 2),
            }
        else:
            bbox = {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "span_x_m": 0.0, "span_y_m": 0.0}

        return cls(
            tier=tier,
            tier_reasons=reasons,
            coreg_result=coreg_result,
            uncertainty=uncertainty,
            multi_start_spread=multi_start_spread,
            spatial_coverage=bbox,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete QC report to a serializable dictionary."""
        return {
            "tier": self.tier,
            "tier_reasons": self.tier_reasons,
            "translation": {
                "dx_px": round(self.coreg_result.dx_px, 4),
                "dy_px": round(self.coreg_result.dy_px, 4),
                "dx_m": round(self.coreg_result.dx_m, 4),
                "dy_m": round(self.coreg_result.dy_m, 4),
                "dz_m": round(self.coreg_result.dz_m, 4),
            },
            "uncertainty": self.uncertainty,
            "residuals": {
                "before": self.coreg_result.stats_before,
                "after": self.coreg_result.stats_after,
                "nmad_improvement_pct": self.coreg_result.nmad_improvement_pct,
            },
            "multi_start_consistency": self.multi_start_spread,
            "spatial_coverage": self.spatial_coverage,
            "optimizer": self.coreg_result.optimizer_result.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self, filepath: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Export report to JSON string or write to file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent, default=str)
        if filepath is not None:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json_str, encoding="utf-8")
        return json_str

    def summary(self) -> str:
        """Human-readable formatted summary string."""
        tier_symbol = "✓ [RELIABLE]" if self.tier == "reliable" else ("! [CAUTIOUS]" if self.tier == "cautious" else "✗ [UNRELIABLE]")
        lines = [
            f"=== GDC Quality Control Report ({tier_symbol}) ===",
            f"QC Tier: {self.tier.upper()}",
            "Reasons / Findings:",
        ]
        for r in self.tier_reasons:
            lines.append(f"  - {r}")
        lines.extend([
            "Translations:",
            f"  dx: {self.coreg_result.dx_m:+.3f} m ({self.coreg_result.dx_px:+.3f} px) ± {self.uncertainty['se_dx_m']:.3f} m",
            f"  dy: {self.coreg_result.dy_m:+.3f} m ({self.coreg_result.dy_px:+.3f} px) ± {self.uncertainty['se_dy_m']:.3f} m",
            f"  dz (vertical bias): {self.coreg_result.dz_m:+.3f} m ± {self.uncertainty['se_dz_m']:.3f} m",
            "Residuals:",
            f"  NMAD: {self.coreg_result.nmad_before:.3f} m -> {self.coreg_result.nmad_after:.3f} m (improvement: {self.coreg_result.nmad_improvement_pct:.1f}%)",
            f"  Valid Points: {self.coreg_result.stats_after.get('count', 0)} / {self.coreg_result.stats_after.get('total_count', 0)}",
            f"Optimizer: {self.coreg_result.optimizer_name} ({self.coreg_result.optimizer_result.nit} iters, {self.coreg_result.optimizer_result.nfev} evals)",
        ])
        return "\n".join(lines)
