# FCEst

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Coverage](https://img.shields.io/badge/coverage-50%25-brightgreen)](coverage.xml)

`FCEst` is a package for estimating static and time-varying functional connectivity (TVFC) in Python.
It includes a range of methods for this task, including Wishart processes, DCC and GO MGARCH models, and sliding windows.

A comparison and benchmarking of these estimation methods can be found [here](https://github.com/OnnoKampman/FCEst-benchmarking).

## Installation

`FCEst` can easily be installed by cloning this repository and using `pip install`:

```zsh
$ git clone https://github.com/OnnoKampman/FCEst.git
$ cd FCEst
$ pip install -e .
```

Make sure you have R installed and that `R_HOME` is set, for example by running `brew install r` on MacOS.

At some point this package will be made directly available from PyPi.

## Usage

Below is a short example demonstrating how to use the high-level API for TVFC estimation.
Additional model demonstrations can be found in Jupyter Notebooks under `.notebooks/Model demos/`.

### Generate dummy data

```python
>>> import numpy as np
>>> N = 200  # number of time steps (scanning volumes)
>>> D = 3  # number of time series (ROIs)
>>> x = np.linspace(0, 1, N).reshape(-1, 1)
>>> y = np.random.random(size=(N, D))
```

### Wishart process

```python
>>> from fcest.models.wishart_process import VariationalWishartProcess
>>> m = VariationalWishartProcess(
        x_observed=x,
        y_observed=y,
    )
>>> tvfc_estimates, tvfc_estimates_stddev = m.predict_corr(x)
```

```python
>>> from fcest.models.wishart_process import SparseVariationalWishartProcess
>>> m = SparseVariationalWishartProcess(
        D=D,
        Z=x,
    )
>>> tvfc_estimates, tvfc_estimates_stddev = m.predict_corr(x)
```

### DCC MGARCH

```python
>>> from fcest.models.mgarch import MGARCH
>>> m = MGARCH(
        mgarch_type="DCC",
    )
>>> m.fit_model(y)
>>> tvfc_estimates = m.predict_corr()
```

### Sliding windows

```python
>>> from fcest.models.sliding_windows import SlidingWindows
>>> sw = SlidingWindows(
        x_train_locations=x,
        y_train_locations=y,
    )
>>> tvfc_estimates = sw.overlapping_windowed_cov_estimation(
        window_length=30,
    )
```

### Sliding windows (with cross-validated window length)

```python
>>> from fcest.models.sliding_windows import SlidingWindows
>>> sw = SlidingWindows(
        x_train_locations=x,
        y_train_locations=y,
    )
>>> cv_window_length = sw.compute_cross_validated_optimal_window_length()
>>> tvfc_estimates = sw.overlapping_windowed_cov_estimation(
        window_length=cv_window_length,
    )
```

### Static functional connectivity

```python
>>> from fcest.models.sliding_windows import SlidingWindows
>>> sw = SlidingWindows(
        x_train_locations=x,
        y_train_locations=y,
    )
>>> sfc_estimates = sw.estimate_static_functional_connectivity()
```

## Contributing

FCEst is an open-source project and contributions from the community are more than welcome.
Please raise an issue on Github or send me a message.

## Curated list of relevant papers

A curated list of publications related to functional connectivity estimation can be found on [Semantic Scholar](https://www.semanticscholar.org/shared/library/folder/8091430).
