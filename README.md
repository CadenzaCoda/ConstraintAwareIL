# CAIL: Constraint-Aware Imitation Learning for Autonomous Racing

**Safety supervision during behavior cloning, without an additional safety filter at deployment.**

Implementation of the experiments in **[A Simple Approach to Constraint-Aware Imitation Learning with Application to Autonomous Racing](https://doi.org/10.1109/IROS60139.2025.11247769)**.

**Shengfan Cao, Eunhyek Joa, Francesco Borrelli**  
2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (**IROS 2025**)

[Paper](https://arxiv.org/abs/2503.07737) · [Published version](https://doi.org/10.1109/IROS60139.2025.11247769) · [Project page](https://cadenzacoda.github.io/portfolio/publications/constraint-aware-imitation-learning/) · [Citation](#citation)

## Overview

CAIL studies **safe imitation learning from imperfect demonstrations that may include unsafe behavior**. It augments behavior cloning with a learned safety penalty, so training accounts for constraint satisfaction alongside imitation accuracy.

The approach is motivated by a differentiable approximation of a minimum-effort predictive safety filter. A learned dynamics model and a learned safe-set membership model provide **privileged safety supervision** using full-state information during training. At deployment, actions come from the learned policy without an additional safety filter.

The paper evaluates autonomous path following and racing in simulation, with both full-state feedback and image-plus-velocity feedback. It reports improved constraint satisfaction and more consistent task performance compared with behavior cloning trained with dataset aggregation.

## How it works

1. **Collect rollouts and expert supervision.** The training loop gathers trajectories using a mixture of expert and learner actions. Demonstrations and rollouts need not all be successful.
2. **Learn a safety critic.** Learn forward dynamics and an approximation of safe-set membership. Safety auto-labeling constructs surrogate labels from successful and failed trajectories, using local convex-hull comparisons to reduce false negative labels. A failed trajectory does not imply that every state along it is unsafe.
3. **Train a constraint-aware policy.** Combine the mean-squared imitation loss with a weighted negative-log-likelihood safety penalty. The safety weight is controlled by `lam` in the model configuration.
4. **Execute the policy.** The learned policy produces actions directly; the dynamics and safety models are used for training rather than online action correction.

See **Section IV, Equation (14), Algorithm 1, and Figure 2** of the [paper](https://arxiv.org/html/2503.07737v2) for the objective and architecture.

### Training and deployment

| Component or information | Training | Policy execution |
| --- | --- | --- |
| Expert policy | Supplies imitation targets and participates in rollout collection | Not needed to produce the learned policy's actions |
| Full-state information | Used for expert and safety supervision | Used by the full-state policy; not required as full state by the image-feedback policy |
| Learned dynamics and safety models | Supply the safety penalty | No additional safety-filter computation |
| Policy observations | Full state, or images plus velocity measurements | The same observation type used to train the policy |

The evaluation harness can query the expert for comparison and logging; the action applied during learned-policy evaluation comes from the learned policy.

### Scope

- **Empirical safety improvements:** the learned policy is not certified to satisfy constraints.
- **Simulation experiments:** physical deployment and sim-to-real transfer are not validated in this paper.
- **Iterative learning:** training includes new rollouts and expert supervision, rather than only a fixed offline dataset.
- **Image-plus-velocity feedback:** `-o camera` includes velocity measurements; it does not mean camera-only sensing.

## Installation

The original experiments were documented with **Python 3.8** and **CARLA 0.9.15**. The bundled Python 3.8 CARLA wheel targets **Linux x86-64**. Use a CARLA Python API build that matches your Python version and operating system if using a different environment.

### 1. Clone the repository and create the environment

```sh
git clone https://github.com/CadenzaCoda/ConstraintAwareIL.git
cd ConstraintAwareIL

conda create -n CAIL python=3.8
conda activate CAIL
```

Run the remaining repository commands from this directory.

### 2. Install Python dependencies and local packages

```sh
python -m pip install -r requirements.txt
python -m pip install -e src/carla_gym/gym-carla
python -m pip install -e src/mpclab_common
python -m pip install -e src/mpclab_controllers
python -m pip install -e src/mpclab_simulation
```

### 3. Install CARLA and its Python API

Download and extract [CARLA 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15). See the [CARLA installation guide](https://carla.readthedocs.io/en/0.9.15/start_quickstart/) for platform requirements.

For the Python 3.8 Linux x86-64 environment above, install the supplied wheel:

```sh
python -m pip install dist/carla-0.9.15-cp38-cp38-linux_x86_64.whl
```

For Python 3.7, the CARLA 0.9.15 distribution provides a corresponding wheel under `PythonAPI/carla/dist`. For other environments, obtain a matching API package or follow the [CARLA build instructions](https://carla.readthedocs.io/en/0.9.15/build_system/). Compatibility with another API wheel alone does not establish compatibility with the repository's remaining dependencies.

### 4. Install HPIPM

The MPCC expert uses **HPIPM** as its quadratic-programming solver. Follow the [HPIPM installation instructions](https://github.com/giaf/hpipm) to build the solver and install its Python interface in the active environment. Follow the upstream instructions for its native dependencies and library paths as well.

## Run the experiments

### Configure the model

The observation mode selects the configuration file automatically:

| Observation option | Configuration | Policy input |
| --- | --- | --- |
| `-o state` | [`config/safeAC.yaml`](config/safeAC.yaml) | Full state |
| `-o camera` | [`config/visionSafeAC.yaml`](config/visionSafeAC.yaml) | RGB image and velocity measurements |

Set `model_hparams.lam` in the selected file before launching a run. Both checked-in configurations currently set `lam` to `1.0`; the full-state racing experiment below calls for `10.0`.

The safety auto-labeling radius `rho` is a separate setting: it defaults to `1.0` in `EfficientReplayBufferPN.preprocess` in [`utils/data_util.py`](utils/data_util.py). It is not currently exposed through the YAML files or a command-line flag. To use a different value, pass it explicitly to the `self.replay_buffer.preprocess(...)` call in `IL_Trainer_CARLA_SafeAC.training_loop` in [`il_trainer.py`](il_trainer.py).

### Start CARLA for image-feedback experiments

In a separate terminal, run the following from the extracted CARLA directory:

```sh
./CarlaUE4.sh -RenderOffScreen -quality-level=Low
```

The trainer connects to `localhost:2000` by default. Use `--host` and `--port` if the server is elsewhere. Full-state experiments disable the camera bridge and do not require a running CARLA server.

### Experiment V-A: Image-feedback path following

Uses a PID expert for conservative-speed path following. Set `lam: 1.0` in `config/visionSafeAC.yaml` and use `rho = 1.0`.

```sh
python il_trainer.py -c pid -o camera -m path_following --n_epochs 50
```

### Experiment V-B: Full-state racing

Uses an MPCC expert for high-speed racing. Set `lam: 10.0` in `config/safeAC.yaml` and use `rho = 1.0`.

```sh
python il_trainer.py -c mpcc-conv -o state -m state_racing --n_epochs 500
```

### Experiment V-C: Image-feedback racing

Uses the MPCC expert with an image-plus-velocity policy. Set `lam: 1.0` in `config/visionSafeAC.yaml`.

**Radius setting:** Figure 7 of arXiv v2 reports `rho = 0.5` for this experiment, while the earlier README specifies `rho = 1.0` and the code defaults to `1.0`. Record which value you use when comparing results. To follow the figure's setting, make the preprocessing call described above `self.replay_buffer.preprocess(rho=0.5)`; restore `rho=1.0` for V-A and V-B.

```sh
python il_trainer.py -c mpcc-conv -o camera -m image_racing --n_epochs 200
```

These commands select the experiment and training budget. They do not automatically apply the paper's checkpoint-selection or early-stopping criterion. The paper evaluates consecutive completed laps, capped at 50, and uses early stopping when a policy completes 50 laps for the second time. Compare checkpoints using the stated evaluation protocol rather than assuming the final training epoch is the best policy.

### Outputs and useful options

The trainer writes logs under `logs/`, model weights under `model_data/`, and selected checkpoints under `model_data/significant_checkpoints/`. Evaluation records completed laps, rewards, lap-time statistics, and trajectory plots.

Use descriptive run names with `-m` so outputs are easy to distinguish. For comparisons, record the repository commit, configuration, `rho`, random seed, and checkpoint-selection rule.

| Option | Purpose |
| --- | --- |
| `-c pid` / `-c mpcc-conv` | Select the expert |
| `-o state` / `-o camera` | Select policy observations |
| `-m NAME` | Label a run and its output files |
| `--n_epochs N` | Set the training budget |
| `--seed N` | Set the NumPy and PyTorch seed; default `42` |
| `--host HOST --port PORT` | Set the CARLA connection |

For the complete command-line interface, run this after installing the dependencies:

```sh
python il_trainer.py -h
```

## Repository guide

| Path | Contents |
| --- | --- |
| [`il_trainer.py`](il_trainer.py) | Rollout collection, training, evaluation, and command-line entry point |
| [`config/`](config/) | Full-state and image-feedback model configurations |
| [`models/safeAC.py`](models/safeAC.py) | Full-state policy, safety critic, and dynamics integration |
| [`models/visionSafeAC.py`](models/visionSafeAC.py) | Image-feedback policy and training losses |
| [`utils/data_util.py`](utils/data_util.py) | Replay buffers and safety auto-labeling |
| [`src/carla_gym/`](src/carla_gym/) | Environment integration and expert controller wrappers |
| [`src/mpclab_common/`](src/mpclab_common/), [`src/mpclab_controllers/`](src/mpclab_controllers/), [`src/mpclab_simulation/`](src/mpclab_simulation/) | Vehicle models, controllers, track data, and simulation utilities |
| [`dist/`](dist/) | Bundled CARLA Python API wheels |

## Citation

If you use this implementation or build on the method, please cite the IROS paper:

```bibtex
@inproceedings{cao2025cail,
  author    = {Cao, Shengfan and Joa, Eunhyek and Borrelli, Francesco},
  title     = {A Simple Approach to Constraint-Aware Imitation Learning with Application to Autonomous Racing},
  booktitle = {2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2025},
  pages     = {9830--9837},
  doi       = {10.1109/IROS60139.2025.11247769}
}
```
