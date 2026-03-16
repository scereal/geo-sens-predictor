"""pyGeo integration layer for applying design variables and sensitivities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .design_variables import apply_modes


@dataclass(slots=True)
class DeformationResult:
    p: np.ndarray  # [n_dv]
    X: np.ndarray  # [n_points,2]
    S: np.ndarray  # [n_points,2,n_dv]
    backend: str


class PyGeoAirfoilWrapper:
    """Wrapper that prefers pyGeo but supports deterministic fallback mode.

    The fallback keeps pipeline runnable in environments where pyGeo is unavailable.
    """

    def __init__(self, points_2d: np.ndarray, ffd_path: str, dv_names: list[str], pygeo_required: bool = True):
        self.points_2d = points_2d
        self.points_3d = np.column_stack([points_2d, np.zeros(points_2d.shape[0])])
        self.ffd_path = ffd_path
        self.dv_names = dv_names
        self.pygeo_required = pygeo_required
        self._dvgeo = None
        self._use_pygeo = False
        self._try_init_pygeo()

    def _try_init_pygeo(self) -> None:
        try:
            from pygeo import DVGeometry  # type: ignore

            dvgeo = DVGeometry(self.ffd_path)
            dvgeo.addPointSet(self.points_3d, "airfoil")
            self._dvgeo = dvgeo
            self._use_pygeo = True
        except Exception as exc:  # noqa: BLE001
            if self.pygeo_required:
                raise RuntimeError(
                    "pyGeo initialization failed. Install pyGeo or set pygeo.required=false in config for fallback mode."
                ) from exc
            warnings.warn(f"Using analytic fallback instead of pyGeo: {exc}", RuntimeWarning)
            self._use_pygeo = False

    def deform_and_sens(self, p: np.ndarray) -> DeformationResult:
        """Apply design vector and return X,S.

        S has shape [n_points,2,n_dv] in final 2D representation.
        """
        p = np.asarray(p, dtype=float)
        if p.ndim != 1 or p.shape[0] != len(self.dv_names):
            raise ValueError("Invalid p shape")

        # Fallback is deterministic and used for tests/debug.
        if not self._use_pygeo:
            X, S = apply_modes(self.points_2d, p, self.dv_names)
            return DeformationResult(p=p, X=X, S=S, backend="analytic-fallback")

        # pyGeo path currently computes deformed coordinates and finite-difference sensitivity.
        # If pyGeo native Jacobian APIs are available in your setup, replace FD section accordingly.
        try:
            assert self._dvgeo is not None
            self._dvgeo.setDesignVars({name: float(val) for name, val in zip(self.dv_names, p, strict=True)})
            x_def_3d = self._dvgeo.update("airfoil")
            X = x_def_3d[:, :2]

            eps = 1e-5
            n_points = X.shape[0]
            n_dv = len(self.dv_names)
            S = np.zeros((n_points, 2, n_dv), dtype=float)
            for i, name in enumerate(self.dv_names):
                p_eps = p.copy()
                p_eps[i] += eps
                self._dvgeo.setDesignVars({n: float(v) for n, v in zip(self.dv_names, p_eps, strict=True)})
                x_eps = self._dvgeo.update("airfoil")[:, :2]
                S[:, :, i] = (x_eps - X) / eps
            self._dvgeo.setDesignVars({name: float(val) for name, val in zip(self.dv_names, p, strict=True)})
            return DeformationResult(p=p, X=X, S=S, backend="pygeo")
        except Exception as exc:  # noqa: BLE001
            if self.pygeo_required:
                raise RuntimeError("pyGeo deformation/sensitivity call failed") from exc
            warnings.warn(f"pyGeo failed at runtime; using fallback modes: {exc}", RuntimeWarning)
            X, S = apply_modes(self.points_2d, p, self.dv_names)
            return DeformationResult(p=p, X=X, S=S, backend="analytic-fallback")
