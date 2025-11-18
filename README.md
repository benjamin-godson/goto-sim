![Python](https://img.shields.io/badge/python-3.9-blue.svg)
[![Run Tests](https://github.com/benjamin-godson/goto-sim/actions/workflows/test.yml/badge.svg)](https://github.com/benjamin-godson/goto-sim/actions/workflows/test.yml)

# Welcome to GOTO Sim

GOTO Sim is a package used to simulate observational strategies for the
Gravitational-wave Optical Transient Observer (GOTO).

## Installation

### Prerequisites
GOTO Sim requires Python 3.9, it may work with other versions but this is the only
version that is tested.

The only dependency that must be manually installed is 
[GOTO-Tile](https://github.com/GOTO-OBS/goto-tile)
which can be installed via pip:

```bash
$ pip install git+https://github.com/GOTO-OBS/goto-tile.git

$ pip install goto-sim
```
## Usage
TODO: Add a brief example of how to use the package to this section

The simplest way to interact with GOTO Sim is to use the Simulator class:

```python
from goto_sim.sim import Simulator
sim = Simulator()
sim.run()
```
This will simulate 24 hours of observations from the current time using default
parameters. See the documentation for more details on how to customize the simulation.

## Contributing


## Copyright

- Copyright © 2025 Ben Godson.
- Free software distributed under the [MIT License](./LICENSE).
