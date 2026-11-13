"""Heat loss of the bar to the surrounding air: natural convection + radiation.

Every outer face that is not pressed against a die loses

    q = h (T - T_air) + eps sigma (T_K^4 - T_sur_K^4)        (outward, per area)

with the air and the surroundings at room temperature (T_air = T_sur = 20 C).
Units of the package: mm, tonne, s, so a heat flux is in mW/mm^2
(1 W/m^2 = 1e-3 mW/mm^2) and h in mW/mm^2-K (1 W/m^2-K = 1e-3 mW/mm^2-K).

Convection coefficient h: natural convection from a horizontal cylinder in
still air, Churchill & Chu (1975) correlation (Incropera, DeWitt, Bergman &
Lavine, Fundamentals of Heat and Mass Transfer, eq. 9.34, valid for
Ra_D <= 1e12):

    Nu_D = {0.60 + 0.387 Ra_D^(1/6) / [1 + (0.559/Pr)^(9/16)]^(8/27)}^2,
    Ra_D = g beta (T_s - T_air) D^3 / (nu alpha),   beta = 1 / T_film (ideal gas),
    h = Nu_D k_air / D,

air properties at the film temperature T_film = (T_s + T_air)/2 from the same
book, Table A.4 (air at 1 atm), linearly interpolated.  Over 400-1000 C the
coefficient changes by only ~10 % (a larger temperature difference against a
thinner, hotter air film), and radiation carries most of the loss at forging
temperature, so h is evaluated once per run at the initial billet temperature
and kept constant.

Emissivity: 0.85, oxidized stainless steel at forging temperature.  Sources:
W. R. Wade, NASA Memo 1-20-59L (1959), total hemispherical emissivity of stably
oxidized type 347 stainless steel 0.87-0.91; IFRF (International Flame Research
Foundation) note on surface emissivity: up to about 0.8 for heavily oxidized
stainless steel.  The value depends on the oxide state and is a run option.

Stefan-Boltzmann constant 5.670374419e-8 W/m^2-K^4 (CODATA 2018)
= 5.670374419e-11 mW/mm^2-K^4.
"""
import numpy as onp

T_AIR = 20.0                      # C, air and surroundings (room temperature)
EMISSIVITY = 0.85
SIGMA_SB = 5.670374419e-11        # mW/mm^2-K^4
KELVIN = 273.15
G = 9.80665                       # m/s^2

# Incropera et al., Table A.4, air at 1 atm:
#   T (K), nu (1e-6 m^2/s), k (1e-3 W/m-K), alpha (1e-6 m^2/s), Pr
_AIR = onp.array([
    [300, 15.89, 26.3, 22.5, 0.707],
    [350, 20.92, 30.0, 29.9, 0.700],
    [400, 26.41, 33.8, 38.3, 0.690],
    [450, 32.39, 37.3, 47.2, 0.686],
    [500, 38.79, 40.7, 56.7, 0.684],
    [550, 45.57, 43.9, 66.7, 0.683],
    [600, 52.69, 46.9, 76.9, 0.685],
    [650, 60.21, 49.7, 87.3, 0.690],
    [700, 68.10, 52.4, 98.0, 0.695],
    [750, 76.37, 54.9, 109.0, 0.702],
    [800, 84.93, 57.3, 120.0, 0.709],
    [850, 93.80, 59.6, 131.0, 0.716],
    [900, 102.9, 62.0, 143.0, 0.720],
    [950, 112.2, 64.3, 155.0, 0.723],
    [1000, 121.9, 66.7, 168.0, 0.726],
])


def air_properties(T_film_K):
    """nu (m^2/s), k (W/m-K), alpha (m^2/s), Pr of air at T_film_K (300-1000 K)."""
    if not (_AIR[0, 0] <= T_film_K <= _AIR[-1, 0]):
        raise ValueError(f"film temperature {T_film_K:.0f} K outside the air table "
                         f"({_AIR[0, 0]:.0f}-{_AIR[-1, 0]:.0f} K)")
    col = [onp.interp(T_film_K, _AIR[:, 0], _AIR[:, j]) for j in range(1, 5)]
    return col[0] * 1e-6, col[1] * 1e-3, col[2] * 1e-6, col[3]


def natural_convection_h(D_mm, T_s, T_air=T_AIR):
    """Churchill-Chu h of a horizontal cylinder of diameter D_mm whose surface is
    at T_s (C) in still air at T_air (C).  Returns (h in mW/mm^2-K, details)."""
    Ts_K, Ta_K = T_s + KELVIN, T_air + KELVIN
    Tf = 0.5 * (Ts_K + Ta_K)
    nu, k, alpha, Pr = air_properties(Tf)
    D = D_mm * 1e-3
    Ra = G * (1.0 / Tf) * abs(Ts_K - Ta_K) * D ** 3 / (nu * alpha)
    if Ra > 1e12:
        raise ValueError(f"Ra_D = {Ra:.3e} outside the Churchill-Chu range (<= 1e12)")
    Nu = (0.60 + 0.387 * Ra ** (1.0 / 6.0) / (1.0 + (0.559 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2
    h_W = Nu * k / D                                   # W/m^2-K
    return h_W * 1e-3, dict(T_film_K=Tf, Ra=Ra, Nu=Nu, h_W=h_W, nu=nu, k=k, alpha=alpha, Pr=Pr)


def radiation_h_equivalent(T, T_sur=T_AIR, eps=EMISSIVITY):
    """eps sigma (T^2 + T_sur^2)(T + T_sur) in W/m^2-K (for reporting only)."""
    a, b = T + KELVIN, T_sur + KELVIN
    return eps * SIGMA_SB * 1e3 * (a * a + b * b) * (a + b)
