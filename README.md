# Materials for pyMOR School 2026

This repository contains course material for the 2026 edition
of [pyMOR School](https://2026.school.pymor.org).

## Installation

To launch a jupyterlab server using [uv](https://docs.astral.sh/uv/),
simply execute

```
uv run jupyter lab
```

This includes the optional pyMOR dependencies used in the lectures: PyTorch and
scikit-learn for the machine learning parts, `slycot` for system-theoretic MOR,
and the Jupyter stack.

## For FEniCSx support

`dolfinx` is not available on PyPI. Use a conda environment instead of the virtualenv above
that is managed by [pixi](https://pixi.prefix.dev/):

```
pixi run jupyter lab
```
