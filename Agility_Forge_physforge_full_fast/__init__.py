## Version 8: Multi-hit forging data generation for VAE training
##
## Based on Version 7 modular refactor, adds:
##   - forging.yml           : YAML config for sequence types, scan, noise
##   - data_storage.py       : SQLite DBMS for training data (start_series, record_hit)
##   - main.py               : YAML-driven data generation loop with noise + DB recording
##
## Unchanged from V7:
##   - hit_config.py          : Hit dataclass
##   - material_data.py       : Temperature-dependent material data (15-5PH)
##   - constitutive.py        : ThermalMechanical Problem class
##   - time_stepper.py        : AutomaticTimeStepperTM adaptive time-stepping
##   - boundary_conditions.py : BC construction, surface integral refresh
##   - mesh_container.py      : MeshContainer for mesh I/O
##
