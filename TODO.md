# To-Do List

## Always there

- [ ] [dev/](./): implementation under development

## Current items

- [ ] [mapper/catalogue_maker.py](../harmonia/mapper/catalogue_maker.py):
      check catalogue fidelity
- [ ] [mapper/spherical_transform.py](../harmonia/mapper/spherical_transform.py):
      1) fix spherical order collapse; 2) generalise for FKP catalogues;
- [ ] [reader/spherical_model.py](../harmonia/reader/spherical_model.py):
      1) fix spherical order collapse; 2) generalise functionals to classes; 3)
      check model computation accuracy (see next below);
- [ ] [algorithms/integration.py](../harmonia/algorithms/integration.py): check
      integration convergence (see previous above)

## On the horizon

- [ ] [mapper/cartesian_reduction.py](../harmonia/mapper/cartesian_reduction.py)
- [ ] [reader/cartesian_model.py](../harmonia/reader/cartesian_model.py)
- [ ] [reader/hybrid.py](../harmonia/reader/hybrid.py)

## Distant future

Incorporate structure growth functions in ``cosmology`` module, and extend
coverage of effects, e.g. evolution, Alcock--Paczynski and integral
constraints.  (RSDs and selection/masking are included in the initial phase.)

## Publishing

- [ ] testing
- [ ] documentation build
- [ ] distribution build
