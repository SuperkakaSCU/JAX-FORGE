"""Hit configuration dataclass for multi-hit forging simulations."""

from dataclasses import dataclass


@dataclass
class Hit:
    """
    Parameters for a single compression hit.

    Parameters
    ----------
    x_min_band : float
        Lower bound of contact band along tool axis (relative to H).
    x_max_band : float
        Upper bound of contact band along tool axis (relative to H).
    compression_displacement : float
        Total platen displacement in this hit (mm).
    rotation_euler_x : float
        Rotation angle around X-axis in degrees.
    total_time : float
        Duration (seconds) allocated for this hit's time-stepping.
    axial_pos_mm : float or None
        [v2] Axial position of the hammer on the *deformed* bar (mm), i.e. the
        raw ``u2`` of the recorded action.  When not None, the caller of
        ``run_forging_series`` re-maps ``x_min_band``/``x_max_band`` onto the
        current deformed length before each hit (see main._remap_band_deformed).
        When None (default) the band fractions are used as-is, which reproduces
        the v1 behaviour exactly.
    band_width_mm : float or None
        [v2] Hammer footprint in mm, measured on the deformed bar.  Only used
        together with ``axial_pos_mm``.
    """
    x_min_band: float
    x_max_band: float
    compression_displacement: float
    rotation_euler_x: float = 0.0
    total_time: float = 1.0
    axial_pos_mm: float = None
    band_width_mm: float = None
