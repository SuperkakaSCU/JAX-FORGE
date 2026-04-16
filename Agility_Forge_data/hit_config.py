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
    """
    x_min_band: float
    x_max_band: float
    compression_displacement: float
    rotation_euler_x: float = 0.0
    total_time: float = 1.0
