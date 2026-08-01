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
    band_center_frac : float or None
        Hammer landing point as a FRACTION OF THE CURRENT (deformed) rod
        length, measured from the current x_min.  Re-mapped onto the reference
        mesh before every hit by ``main._remap_band_current``.  None keeps the
        legacy behaviour (x_min_band/x_max_band taken as fractions of the
        REFERENCE length H, i.e. the band no longer tracks the rod once it
        elongates).
    band_width_mm : float or None
        PHYSICAL hammer width in mm, constant for the whole run — the die is
        steel and does not widen when the rod stretches.  The legacy
        ``band_width`` was a fraction of H, so a 0.2 band on a rod that
        elongates 50% would silently cover 50% more material.
    """
    x_min_band: float
    x_max_band: float
    compression_displacement: float
    rotation_euler_x: float = 0.0
    total_time: float = 1.0
    band_center_frac: float = None
    band_width_mm: float = None
