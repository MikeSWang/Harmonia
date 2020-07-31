.. title:: Harmonia Documentation

============================================================
**Hybrid-Basis Inference for Large-Scale Galaxy Clustering**
============================================================

.. image:: https://img.shields.io/badge/arXiv-2007.14962-important
    :target: https://arxiv.org/abs/2007.14962
    :alt: arXiv eprint
.. image:: https://img.shields.io/github/v/release/MikeSWang/Harmonia?label=release
    :target: https://github.com/MikeSWang/Harmonia/releases/latest
    :alt: GitHub release (latest by date)
.. image:: https://readthedocs.org/projects/harmonia/badge/?version=latest
    :target: https://harmonia.readthedocs.io/en/latest
    :alt: Documentation status
.. image:: https://travis-ci.com/MikeSWang/Harmonia.svg?branch=master
    :target: https://travis-ci.com/MikeSWang/Harmonia
    :alt: Build status
.. image:: https://img.shields.io/badge/licence-GPLv3-informational
    :target: https://github.com/mikeswang/Harmonia/tree/master/LICENCE
    :alt: Licence

|Harmonia| is a Python package
that combines clustering statistics decomposed in spherical and Cartesian
Fourier bases for large-scale galaxy clustering likelihood analysis.

.. |Harmonia| raw:: html

    <span style="font-variant: small-caps">Harmonia</span>


Installation
============

We recommend that you install |nbodykit|_ first by following these
`instructions <https://nbodykit.readthedocs.io/en/latest/getting-started/install.html>`_.

After that, you can install |Harmonia| simply using ``pip``::

    pip install harmoniacosmo

Note that only here does the name ``harmoniacosmo`` appear because
unfortunately on PyPI the project name ``harmonia`` has already been taken.

.. |nbodykit| replace:: ``nbodykit``
.. _nbodykit: https://nbodykit.readthedocs.io/en/latest


Documentation
=============

- :doc:`recipes` (under construction): tutorials in the format of integrated
  notebooks showcasing the use of |Harmonia| will be gradually added, so look
  out for any updates! For now, |application|_ offers some scripts that
  illustrate the use of |Harmonia|.
- :doc:`apidoc`: more detailed documentation of classes and functions.

.. |application| replace:: ``application``
.. _application: https://github.com/MikeSWang/Harmonia/tree/master/application


Attribution
===========

If you would like to acknowledge this work, please cite
`Wang et al. (2020) <https://arxiv.org/abs/2007.14962>`_. You
may use the following BibTeX record::

    @article{Wang_2020b,
        author={Wang, M.~S. and Avila, S. and Bianchi, D. and Crittenden, R. and Percival, W.~J.},
        title={Hybrid-basis inference for large-scale galaxy clustering: combining spherical and {Cartesian} {Fourier} analyses},
        year={2020},
        eprint={2007.14962},
        archivePrefix={arXiv},
        primaryClass={astro-ph.CO},
    }


Licence
=======

|Harmonia| is made freely available under the `GPL v3.0
<https://www.gnu.org/licenses/gpl-3.0.en.html>`_ licence.


.. toctree::
    :hidden:

    recipes
    apidoc
