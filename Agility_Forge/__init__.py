## Version 7: Modular refactor of jax_forge_hits.py (Version 6)
##
## Splits the monolithic file into separate modules for maintainability:
##   - hit_config.py          : Hit dataclass for multi-hit simulation parameters
##   - material_data.py       : Temperature-dependent material data tables (15-5PH)
##   - constitutive.py        : ThermalMechanical Problem class (J2 plasticity, weak form)
##   - time_stepper.py        : AutomaticTimeStepperTM adaptive time-stepping class
##   - boundary_conditions.py : BC construction, surface integral refresh utilities
##   - main.py                : Entry-point driver for multi-hit forging simulation
##
## All Version 2–6 features are preserved. No functional changes.
