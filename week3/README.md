# TAC450 Week3 Building Neural Networks

### About
The material in this repo contains teaching materials related to building neural networks for deep learning applications. 

### Software Environment Setup
First login to CARC OnDemand: https://ondemand.carc.usc.edu/ and request a 'Discovery Cluster Shell Access' within OpenOnDemand. 

We will use Conda to build software packages. We have prepared a setup script for installation. 

First, we need to request an interactive session. 

If the reservation is on Tuesday:
```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32GB --time=01:00:00 --account=irahbari_1147 --reservation=tac450-tu
```

If the reservation is on Thursday:
```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32GB --time=01:00:00 --account=irahbari_1147 --reservation=tac450-th
```

If the reservation is not available, please use the following command to request an interactive session: 
```bash
salloc --partition=gpu --gres=gpu:v100:1 --cpus-per-task=8 --mem=32GB --time=01:00:00 --account=irahbari_1147
```


change your working direcotry to your scratch directory:
```
cd /scratch1/$(whoami)
```
```
mkdir week3
cd week3
```
```
git clone https://github.com/uschpc/TAC450-DataScience-Fall2026
cd TAC450-DataScience-Fall2026/week3
```
```
module load conda
```
```
python carc_conda_setup.py
```
Note: if after running this python carc_conda_setup.py script, if your terminal still shows something like 
```
bash4-4$
```
Please exit from the interactive session and request another new interactive session to refresh the settings, this way your terminal will look normal (show your compute node e.g. a01-01)

Test PyTorch & line_profiler is property installed: 
```
$ conda activate torch-env
(torch-env) $ python
>> import torch
>> import line_profiler
>> exit()
```

Note: 

**'line_profiler'** is a Python module used for profiling (measuring the execution time) of individual lines in a script, helping with performance optimization.


### Install Jupyter Kernel

A Jupyter kernel is a computational engine that executes the code contained in Jupyter notebooks. Each notebook is connected to a specific kernel, which runs the code in the programming language chosen by the user.

For example:

If you’re working in a Python notebook, it will be connected to a Python kernel, allowing Python code execution.
Similarly, Jupyter supports kernels for other languages, such as R, Julia, and MATLAB.

The kernel manages the state of the notebook (such as variables, imports, and output), allowing you to run cells independently while maintaining continuity across the notebook. You can select or switch kernels from within the Jupyter interface.






