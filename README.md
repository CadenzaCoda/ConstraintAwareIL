Implementation of experiments in "*A Simple Approach to Constraint-Aware Imitation Learning with Application to Autonomous Racing*" (IROS 2025).

# Getting Started
## Python
```shell
conda create -n CAIL python=3.8
conda activate CAIL
pip install -e src/carla-gym/gym_carla
pip install -e src/mpclab_common
pip install -e src/mpclab_controllers
pip install -e src/mpclab_simulation
pip install -r requirements.txt
```
Note that we only tested the implementation on Python 3.8. 

## CARLA
This implementation relies on CARLA for camera images. 

Follow official documentation for CARLA for installation guide for your OS. 
The easiest way to install is to download and unzip the precompiled version of CARLA for your OS. 
We used [CARLA 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15) in our simulations. 

You also need to install CARLA's Python API into your Python environment. 
If you choose to use the precompiled version, do the following after unzipping. 
```shell
cd $CARLA_ROOT/PythonAPI/carla/dist
pip install carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
```
Replace `$CARLA_ROOT` with the root directory of CARLA. 
As implied by the file name, you must use Python 3.7 to be compatible with the provided `.whl` file, which means you may 
potentially run into some compatibility issues that we didn't encounter.  
However, if you built CARLA from source, you may also follow the official documentation to build the Python API for your
Python version. 

# Running the experiments
Before running the experiments: 
- Set the model hyperparameters in `config/safeAC.yaml` (for full-state feedback experiments) or `config/visionSafeAC.yaml` (for image feedback experiments). See the example files for template.
- Start the CARLA server if running image feedback experiments. For enhanced reliability, we recommend running with the following flags: `-RenderOffScreen -quality-level=Low`.

For each experiment in the paper, run the following. 

## Experiment V-A: Image Feedback Autonomous Path Following
```shell
python il_trainer.py -c pid -o img -m <comment_for_logs> --n_epochs 50 
```

## Experiment V-B: Full-state Feedback Autonomous Car Racing
```shell
python il_trainer.py -c mpcc-conv -o state -m <comment_for_logs> --n_epochs 500
```

## Experiment V-C: Image Feedback Autonomous Car Racing 
```shell
python il_trainer.py -c mpcc-conv -o img -m <comment_for_logs> --n_epochs 200
```