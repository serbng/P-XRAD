from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Union


class BackgroundMethod(str, Enum):
    """
    Background estimation strategy.

    Notes
    -----
    - USER_INPUT is a *metadata* mode: it declares that the background used in the
      pipeline is provided externally at runtime (as a numpy array), and therefore
      is not expected to be stored inside the final YAML results.
    """
    MEDIAN_FILTER = "MEDIAN_FILTER"
    GAUSSIAN_BLUR = "GAUSSIAN_BLUR"
    PERCENTILE_FILTER = "PERCENTILE_FILTER"
    USER_INPUT = "USER_INPUT"


class ThresholdType(str, Enum):
    """Thresholding mode applied to the background-subtracted image."""
    NSIGMA = "NSIGMA"
    ABSOLUTE = "ABSOLUTE"


class RefinementModel(str, Enum):
    """Parametric model used in peak refinement (non-linear fitting stage)."""
    GAUSS_ELLIPTIC = "GAUSS_ELLIPTIC"
    PSEUDO_VOIGT_ELLIPTIC = "PSEUDO_VOIGT_ELLIPTIC"


class BackgroundModel(str, Enum):
    """
    Local background model used during refinement fits.

    CONSTANT
        Constant offset in the ROI.
    PLANE
        Affine plane in the ROI (recommended as default).
    """
    CONSTANT = "CONSTANT"
    PLANE = "PLANE"


@dataclass(frozen=True, slots=True)
class BackgroundConfig:
    """
    Background estimation configuration.

    Parameters
    ----------
    method
        Background estimation method.
    size
        Window size (in pixels) for window-based methods (median/percentile).
        If an even value is provided, implementations may internally bump it to
        the next odd value.
    sigma
        Standard deviation (in pixels) for GAUSSIAN_BLUR.
    percentile
        Percentile (0..100) for PERCENTILE_FILTER.

    Notes
    -----
    In USER_INPUT mode, all numeric parameters are ignored by the estimator.
    """
    method: BackgroundMethod = BackgroundMethod.MEDIAN_FILTER
    size: int = 31
    sigma: Optional[float] = None
    percentile: Optional[float] = None

    def validate(self) -> None:
        if int(self.size) <= 0:
            raise ValueError(f"background.size must be > 0, got {self.size}.")
        if self.sigma is not None and float(self.sigma) <= 0:
            raise ValueError(f"background.sigma must be > 0, got {self.sigma}.")
        if self.percentile is not None and not (0.0 <= float(self.percentile) <= 100.0):
            raise ValueError(f"background.percentile must be in [0,100], got {self.percentile}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "size": int(self.size),
            "sigma": None if self.sigma is None else float(self.sigma),
            "percentile": None if self.percentile is None else float(self.percentile),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BackgroundConfig":
        cfg = cls(
            method=BackgroundMethod(d.get("method", BackgroundMethod.MEDIAN_FILTER.value)),
            size=int(d.get("size", 31)),
            sigma=d.get("sigma", None),
            percentile=d.get("percentile", None),
        )
        cfg.validate()
        return cfg


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    """
    Thresholding configuration for candidate peak detection.

    The threshold is applied to the *background-subtracted* image (high-pass).

    Parameters
    ----------
    type
        NSIGMA or ABSOLUTE.
    nsigma
        If type == NSIGMA, threshold = nsigma * robust_sigma(highpass),
        where robust_sigma is estimated from MAD (Median Absolute Deviation).
    absolute
        If type == ABSOLUTE, threshold is a fixed absolute value.
    positive_only
        If True, the high-pass image is clipped to be non-negative
        before thresholding (common for diffraction spots).
    """
    type: ThresholdType = ThresholdType.NSIGMA
    nsigma: float = 6.0
    absolute: Optional[float] = None
    positive_only: bool = True

    def validate(self) -> None:
        if self.type == ThresholdType.NSIGMA:
            if float(self.nsigma) <= 0:
                raise ValueError(f"threshold.nsigma must be > 0, got {self.nsigma}.")
        elif self.type == ThresholdType.ABSOLUTE:
            if self.absolute is None:
                raise ValueError("threshold.absolute must be set when threshold.type=ABSOLUTE.")
            if float(self.absolute) <= 0:
                raise ValueError(f"threshold.absolute must be > 0, got {self.absolute}.")
        else:
            raise ValueError(f"Unknown threshold.type: {self.type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "nsigma": float(self.nsigma),
            "absolute": None if self.absolute is None else float(self.absolute),
            "positive_only": bool(self.positive_only),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ThresholdConfig":
        cfg = cls(
            type=ThresholdType(d.get("type", ThresholdType.NSIGMA.value)),
            nsigma=float(d.get("nsigma", 6.0)),
            absolute=d.get("absolute", None),
            positive_only=bool(d.get("positive_only", True)),
        )
        cfg.validate()
        return cfg


@dataclass(frozen=True, slots=True)
class BlobsConfig:
    """
    Filters applied to connected components (blobs) during detection.

    Parameters
    ----------
    min_area
        Minimum number of pixels in a blob to be considered a valid peak.
    max_area
        Optional maximum area to discard very large blobs (streaks, saturation,
        merged peaks, artifacts).
    dist_from_border
        Discard blobs whose bounding box touches the image border closer than
        this distance (in pixels). Useful to avoid truncated peaks.
    """
    min_area: int = 3
    max_area: Optional[int] = None
    dist_from_border: int = 2

    def validate(self) -> None:
        if int(self.min_area) <= 0:
            raise ValueError(f"blobs.min_area must be > 0, got {self.min_area}.")
        if self.max_area is not None:
            if int(self.max_area) <= 0:
                raise ValueError(f"blobs.max_area must be > 0, got {self.max_area}.")
            if int(self.max_area) < int(self.min_area):
                raise ValueError("blobs.max_area must be >= blobs.min_area.")
        if int(self.dist_from_border) < 0:
            raise ValueError(f"blobs.dist_from_border must be >= 0, got {self.dist_from_border}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_area": int(self.min_area),
            "max_area": None if self.max_area is None else int(self.max_area),
            "dist_from_border": int(self.dist_from_border),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BlobsConfig":
        max_area = d.get("max_area", None)
        cfg = cls(
            min_area=int(d.get("min_area", 3)),
            max_area=None if max_area is None else int(max_area),
            dist_from_border=int(d.get("dist_from_border", 2)),
        )
        cfg.validate()
        return cfg


@dataclass(frozen=True, slots=True)
class LocalMaxConfig:
    """
    Local-maxima filtering configuration.

    During detection, candidates are restricted to pixels that are local maxima
    within a neighborhood. This reduces blob merging for nearby peaks.

    Parameters
    ----------
    size
        Neighborhood size (pixels). If even, implementations may internally bump
        to the next odd.
    """
    size: int = 3

    def validate(self) -> None:
        if int(self.size) < 1:
            raise ValueError(f"local_max.size must be >= 1, got {self.size}.")

    def to_dict(self) -> dict[str, Any]:
        return {"size": int(self.size)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LocalMaxConfig":
        cfg = cls(size=int(d.get("size", 3)))
        cfg.validate()
        return cfg


@dataclass(frozen=True, slots=True)
class DedupConfig:
    """
    De-duplication and output limiting configuration.

    Parameters
    ----------
    min_distance
        If > 0, apply non-maximum suppression in (u, v) space to remove peaks
        closer than this distance (in pixels).
    max_peaks
        If set, keep only the top-N peaks by a score (typically I_max).
    """
    min_distance: float = 0.0
    max_peaks: Optional[int] = None

    def validate(self) -> None:
        if float(self.min_distance) < 0:
            raise ValueError(f"dedup.min_distance must be >= 0, got {self.min_distance}.")
        if self.max_peaks is not None and int(self.max_peaks) <= 0:
            raise ValueError(f"dedup.max_peaks must be > 0, got {self.max_peaks}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_distance": float(self.min_distance),
            "max_peaks": None if self.max_peaks is None else int(self.max_peaks),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DedupConfig":
        max_peaks = d.get("max_peaks", None)
        cfg = cls(
            min_distance=float(d.get("min_distance", 0.0)),
            max_peaks=None if max_peaks is None else int(max_peaks),
        )
        cfg.validate()
        return cfg


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    """
    Peak refinement (fitting) configuration.

    Refinement is optional and is typically applied only to the top-N peaks by
    intensity. This stage fits a parametric model on a small ROI around each
    detected peak (sub-pixel positions, widths, background, etc.).

    Parameters
    ----------
    enabled
        Enable refinement stage.
    model
        Parametric peak model (Gaussian / pseudo-Voigt).
    max_k
        Maximum number of peaks per ROI (for overlapped-peak refinement).
        (Implementation may start with max_k=1, then add >1 later.)
    topn
        Refine only the top-N detected peaks (by I_max or similar).
    roi_half_size
        Half-size of square ROI in pixels. ROI size is (2*roi_half_size + 1).
    background_model
        Local background model used during fit.
    """
    enabled: bool = False
    model: RefinementModel = RefinementModel.GAUSS_ELLIPTIC
    max_k: int = 2
    topn: int = 500
    roi_half_size: int = 7
    background_model: BackgroundModel = BackgroundModel.PLANE

    def validate(self) -> None:
        if int(self.max_k) < 1:
            raise ValueError(f"refinement.max_k must be >= 1, got {self.max_k}.")
        if int(self.topn) <= 0:
            raise ValueError(f"refinement.topn must be > 0, got {self.topn}.")
        if int(self.roi_half_size) < 2:
            raise ValueError(f"refinement.roi_half_size must be >= 2, got {self.roi_half_size}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "model": self.model.value,
            "max_k": int(self.max_k),
            "topn": int(self.topn),
            "roi_half_size": int(self.roi_half_size),
            "background_model": self.background_model.value,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RefinementConfig":
        cfg = cls(
            enabled=bool(d.get("enabled", False)),
            model=RefinementModel(d.get("model", RefinementModel.GAUSS_ELLIPTIC.value)),
            max_k=int(d.get("max_k", 2)),
            topn=int(d.get("topn", 500)),
            roi_half_size=int(d.get("roi_half_size", 7)),
            background_model=BackgroundModel(d.get("background_model", BackgroundModel.PLANE.value)),
        )
        cfg.validate()
        return cfg


@dataclass(frozen=True, slots=True)
class PeakSearchConfig:
    """
    Full configuration for peak search: detection + optional refinement.

    Conventions
    -----------
    The peak search module follows standard image conventions:

    - Image indexing is img[v, u] = img[row, col]
    - u increases to the right (columns), v increases downward (rows)
    - Returned coordinates (u, v) are float pixel coordinates (0-based),
      representing peak centers (sub-pixel).

    Debugging
    ---------
    If debugging is True, pipeline functions are allowed to return additional
    intermediate arrays (background, masks, labels, etc.) to help inspecting
    the algorithm in a notebook. Debugging must not be enabled by default to
    avoid performance and memory overhead in large raster scans.

    Practical usage
    ---------------
    For end users, prefer the builder constructor:

        cfg = PeakSearchConfig.build(
            background_method="MEDIAN_FILTER",
            background_size=31,
            nsigma=6.0,
            min_area=3,
            debugging=True,
        )

    YAML schema
    -----------
    This config maps naturally to a nested YAML structure under:

        peaksearch:
          config:
            background: ...
            threshold: ...
            blobs: ...
            local_max: ...
            dedup: ...
            refinement: ...
            debugging: ...
    """
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    blobs: BlobsConfig = field(default_factory=BlobsConfig)
    local_max: LocalMaxConfig = field(default_factory=LocalMaxConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    debugging: bool = False

    def validate(self) -> None:
        self.background.validate()
        self.threshold.validate()
        self.blobs.validate()
        self.local_max.validate()
        self.dedup.validate()
        self.refinement.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "background": self.background.to_dict(),
            "threshold": self.threshold.to_dict(),
            "blobs": self.blobs.to_dict(),
            "local_max": self.local_max.to_dict(),
            "dedup": self.dedup.to_dict(),
            "refinement": self.refinement.to_dict(),
            "debugging": bool(self.debugging),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PeakSearchConfig":
        cfg = cls(
            background=BackgroundConfig.from_dict(d.get("background", {})),
            threshold=ThresholdConfig.from_dict(d.get("threshold", {})),
            blobs=BlobsConfig.from_dict(d.get("blobs", {})),
            local_max=LocalMaxConfig.from_dict(d.get("local_max", {})),
            dedup=DedupConfig.from_dict(d.get("dedup", {})),
            refinement=RefinementConfig.from_dict(d.get("refinement", {})),
            debugging=bool(d.get("debugging", False)),
        )
        cfg.validate()
        return cfg

    @classmethod
    def build(
        cls,
        *,
        background_method: Union[str, BackgroundMethod] = BackgroundMethod.MEDIAN_FILTER,
        background_size: int = 31,
        background_sigma: Optional[float] = None,
        background_percentile: Optional[float] = None,
        threshold_type: Union[str, ThresholdType] = ThresholdType.NSIGMA,
        nsigma: float = 6.0,
        absolute: Optional[float] = None,
        positive_only: bool = True,
        min_area: int = 3,
        max_area: Optional[int] = None,
        dist_from_border: int = 2,
        local_max_size: int = 3,
        min_distance: float = 0.0,
        max_peaks: Optional[int] = None,
        debugging: bool = False,
        refine_enabled: bool = False,
        refine_model: Union[str, RefinementModel] = RefinementModel.GAUSS_ELLIPTIC,
        refine_max_k: int = 2,
        refine_topn: int = 500,
        refine_roi_half_size: int = 7,
        refine_background_model: Union[str, BackgroundModel] = BackgroundModel.PLANE,
    ) -> "PeakSearchConfig":
        """
        Convenience constructor for end users.

        This avoids manually constructing nested dicts or nested dataclasses.

        Parameters
        ----------
        background_method, background_size, background_sigma, background_percentile
            Background estimation settings.
        threshold_type, nsigma, absolute, positive_only
            Thresholding settings on the background-subtracted image.
        min_area, max_area, dist_from_border
            Blob filtering settings.
        local_max_size
            Neighborhood size for local-maximum filtering.
        min_distance, max_peaks
            De-duplication and limiting of output peaks.
        debugging
            If True, allow returning intermediate arrays for inspection.
        refine_*
            Refinement settings (fitting). Can be enabled later.

        Returns
        -------
        PeakSearchConfig
            Validated configuration instance.
        """
        bg_m = background_method if isinstance(background_method, BackgroundMethod) else BackgroundMethod(str(background_method))
        th_t = threshold_type if isinstance(threshold_type, ThresholdType) else ThresholdType(str(threshold_type))
        rf_m = refine_model if isinstance(refine_model, RefinementModel) else RefinementModel(str(refine_model))
        rf_bg = refine_background_model if isinstance(refine_background_model, BackgroundModel) else BackgroundModel(str(refine_background_model))

        cfg = cls(
            background=BackgroundConfig(
                method=bg_m,
                size=int(background_size),
                sigma=background_sigma,
                percentile=background_percentile,
            ),
            threshold=ThresholdConfig(
                type=th_t,
                nsigma=float(nsigma),
                absolute=absolute,
                positive_only=bool(positive_only),
            ),
            blobs=BlobsConfig(
                min_area=int(min_area),
                max_area=None if max_area is None else int(max_area),
                dist_from_border=int(dist_from_border),
            ),
            local_max=LocalMaxConfig(size=int(local_max_size)),
            dedup=DedupConfig(
                min_distance=float(min_distance),
                max_peaks=None if max_peaks is None else int(max_peaks),
            ),
            refinement=RefinementConfig(
                enabled=bool(refine_enabled),
                model=rf_m,
                max_k=int(refine_max_k),
                topn=int(refine_topn),
                roi_half_size=int(refine_roi_half_size),
                background_model=rf_bg,
            ),
            debugging=bool(debugging),
        )
        cfg.validate()
        return cfg