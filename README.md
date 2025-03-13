# Constraint-Aware Imitation Learning for Autonomous Racing

This repository contains the implementation of experiments from [*"A Simple Approach to Constraint-Aware Imitation Learning with Application to Autonomous Racing"*](https://arxiv.org/abs/2503.07737) (Submitted to IEEE IROS 2025).

---

## Prerequisites

### Python Environment Setup
We recommend using **Python 3.8**, as this implementation has only been tested on it.

```sh
conda create -n CAIL python=3.8
conda activate CAIL
pip install -r requirements.txt
pip install -e src/carla_gym/gym-carla
pip install -e src/mpclab_common
pip install -e src/mpclab_controllers
pip install -e src/mpclab_simulation
```

### CARLA Installation
This implementation relies on **CARLA** for camera-based experiments.

- Follow the [CARLA official installation guide](https://carla.readthedocs.io/en/latest/start_quickstart/) for your OS. The simplest way is to download and unzip the precompiled version.
- We used **CARLA 0.9.15** in our simulations. You can download it [here](https://github.com/carla-simulator/carla/releases/tag/0.9.15).

#### Installing CARLA's Python API
For proper simulation functionality, install CARLA's Python API based on your Python version:

- **Python 3.8 (Recommended)**: Use the provided `.whl` file.
  ```sh
  pip install dist/carla-0.9.15-cp38-cp38-linux_x86_64.whl
  ```
- **Python 3.7**: If using the precompiled version, run the following after unzipping:
  ```sh
  cd $CARLA_ROOT/PythonAPI/carla/dist
  pip install carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
  ```
  Replace `$CARLA_ROOT` with the CARLA installation directory.
- **Other Python Versions**: You must build CARLA from source and generate the Python API accordingly. Follow the [official build instructions](https://carla.readthedocs.io/en/latest/build_system/).

### HPIPM Installation
The expert policy in this repository uses **HPIPM** as the optimization solver.

Follow the [HPIPM installation guide](https://github.com/giaf/hpipm) to set up the environment and install its Python API in your Python environment.

---

## Running the Experiments
Before running any experiment:
- Configure the model hyperparameters in:
  - `config/safeAC.yaml` (for **full-state feedback** experiments)
  - `config/visionSafeAC.yaml` (for **image feedback** experiments)
- If running **image feedback experiments**, start the CARLA server. For better reliability, run CARLA with the following flags:
  ```sh
  ./CarlaUE4.sh -RenderOffScreen -quality-level=Low
  ```

### Running Specific Experiments
Each experiment corresponds to a section in the paper. Run the following commands:

#### **Experiment V-A: Image Feedback Autonomous Path Following**
```sh
python il_trainer.py -c pid -o camera -m <comment_for_logs> --n_epochs 50
```

#### **Experiment V-B: Full-State Feedback Autonomous Car Racing**
```sh
python il_trainer.py -c mpcc-conv -o state -m <comment_for_logs> --n_epochs 500
```

#### **Experiment V-C: Image Feedback Autonomous Car Racing**
```sh
python il_trainer.py -c mpcc-conv -o camera -m <comment_for_logs> --n_epochs 200
```

### Additional Help
For a full list of available command-line arguments, run:
```sh
python il_trainer.py -h
```
