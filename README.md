# Density Constrained Reinforcement Learning

The repository constains an official implementation of **Density Constrained Reinforcement Learning**.

### Install
Clone this repository
```bash
git clone https://github.com/Zengyi-Qin/dcrl.git
```

Create an virtual environment with Python 3.6 using Anaconda:
```bash
conda create - n dcrl python=3.6
conda activate dcrl
```

Install the dependencies:
```bash
pip install - r requirements.txt
```

Add `dcrl` to your `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:'path/to/dcrl'
```

## Training
```bash
python examples/FILE --mode train --constrained 1 --output OUTPUT_DIR
```
|           Environment options         |        FILE         |
| --------------------------------------| ------------------- |
| Autonomous electric vehicle routing   |     ddpg_aev.py     |
| Agricultural pesticide spraying drone |     ddpg_farm.py    |
| Direct current series motor control   |    ddpg_motor.py    |

The `--constrained` flag has three options. 0 for no constraint, 1 for the proposed DCRL approach and 2 for the RCPO approach.

## Testing
```bash
python examples/FILE --mode test --constrained 1 --output OUTPUT_DIR --weights WEIGHTS_PATH
```
The `WEIGHTS_PATH` points to the weight files saved in `OUTPUT_DIR/weights` during training. For example, if we run the training of autonomous electric vehicle routing, the weight file can be `OUTPUT_DIR/weights/ddpg_aev_50.h5f`.

To test all the weights in the output foloer, run:
``bash
python examples/FILE --mode test_all --constrained 1 --output OUTPUT_DIR --weights OUTPUT_DIR/weights
```