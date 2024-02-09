# FCEst

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

Methods for estimating time-varying functional connectivity (TVFC).

## Installation

`FCEst` can easily be installed by cloning this repository and using `pip install`:

```zsh
git clone https://github.com/OnnoKampman/FCEst.git
cd FCEst
pip install -e .
```

## Usage

Below is a short example demonstrating how to use the high-level API for TVFC estimation.

### Generate dummy data

```python
>>> import numpy as np
>>> import pandas as pd
>>> N = 100  # number of time steps (scanning volumes)
>>> D = 3  # number of time series (ROIs)
>>> x = np.linspace(0, 1, N)
>>> y = np.random.random(size=(N, D))
>>> df = pd.DataFrame(y)
```

### Wishart process

```python
>>> from fcest.models.wishart_process import VariationalWishartProcess
>>> m = VariationalWishartProcess(
        x_observed=x,
        y_observed=y,
    )
>>> tvfc_estimates = m.predict_corr(x)
```

### DCC MGARCH

```python
>>> from fcest.models.mgarch import MGARCH
>>> dcc = MGARCH(
        mgarch_type="DCC",
    )
>>> dcc.fit_model(df)
>>> tvfc_estimates = dcc.train_location_covariance_structure
```

### Sliding windows

```python
>>> from fcest.models.sliding_windows import SlidingWindows
>>> sw = SlidingWindows(
        x_train_locations=x,
        y_train_locations=y,
    )
>>> tvfc_estimates = sliding_windows.overlapping_windowed_cov_estimation(
        window_length=30,
    )
```

## Contributing

FCEst is an open-source project and contributions from the community are more than welcome.
Please raise an issue on Github or send me a message.

## Curated list of relevant papers

[Semantic Scholar](https://www.semanticscholar.org/shared/library/folder/8091430)
