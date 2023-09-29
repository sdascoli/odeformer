[![PyPI](https://img.shields.io/pypi/v/odeformer.svg)](
https://pypi.org/project/odeformer/)
[![Colab](https://img.shields.io/badge/colab-notebook-yellow)](https://colab.research.google.com/github/sdascoli/odeformer/blob/main/ODEFormer_demo.ipynb)


# ODEformer: symbolic regression of dynamical systems with transformers

This repository contains code for the paper [ODEformer: symbolic regression of dynamical systems with transformers]().

## Installation
This package is installable via pip:

```pip install odeformer```

## Demo

We include a small notebook that loads a pre-trained model you can play with:
[![Colab](https://img.shields.io/badge/colab-notebook-yellow)](https://colab.research.google.com/github/sdascoli/odeformer/blob/main/ODEFormer_demo.ipynb)

## Usage

Import the model in a few lines of code:
```python
import odeformer
from odeformer.model import SymbolicTransformerRegressor
dstr = SymbolicTransformerRegressor(from_pretrained=True)
model_args = {'beam_size':50, 'beam_temperature':0.1}
dstr.set_model_args(model_args)
```

Basic usage:
```python
import numpy as np
from odeformer.metrics import r2_score

times = np.linspace(0, 10, 50)
x = 2.3*np.cos(times+.5)
y = 1.2*np.sin(times+.1)
trajectory = np.stack([x, y], axis=1)

dstr.fit(times, trajectory)
dstr.print_predictions(n_candidates=1)
pred_trajectory = dstr.predict(times, trajectory[0])
print(r2_score(trajectory, pred_trajectory))
```


## Training and evaluation

To launch a model training with additional arguments (arg1,val1), (arg2,val2):
```python train.py --arg1 val1 --arg2 --val2```

All hyper-parameters related to training are specified in ```parsers.py```, and those related to the environment are in ```envs/environment.py```.

To launch evaluation, please use the flag ```reload_checkpoint``` to specify in which folder the saved model is located:
```python evaluate.py --reload_checkpoint XXX```


## License

This repository is licensed under MIT licence.
