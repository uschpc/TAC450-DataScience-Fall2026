# TAC week5 

Tensorboard Tutorial Link: https://pytorch.org/tutorials/beginner/introyt/tensorboardyt_tutorial.html


## Software Environment Setup

First login to CARC OnDemand: https://ondemand.carc.usc.edu/ and request a 'Discovery Cluster Shell Access' within OpenOnDemand. 

We will use Conda to build software packages. If it is the first time you are using Conda, make sure you follow the guide of week3. 

We need to request an interactive session:

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
To install Tensorboard: 

```
conda activate torch-env
conda install tensorboard
```

Change your working direcotry to your scratch directory and clone the repo:
```
cd /scratch1/$(whoami)
```
```
mkdir week5
cd week5
```
```
git clone https://github.com/uschpc/TAC450-DataScience-Fall2025
cd TAC450-DataScience-Fall2025/week5
```

After the installation, please exit from the compute node. 
```
exit
```

## Run the Jupyter Notebook 
Launch the Jupyter Notebook using OnDemand JupyterLab and run the different code blocks within the jupyter notebook. 


## Tensorboard

TensorBoard is a powerful visualization tool. It provides developers and researchers with insights into their machine learning models by allowing them to visualize metrics, model graphs, and other key aspects of their workflows. Whether you’re debugging a model, tuning hyperparameters, or simply seeking a better understanding of your training process, TensorBoard offers a suite of features to facilitate these tasks.

Some key features of Tensorboard include: 

1 Showing images in TensorBoard

2 Graphing scalers and metrics to visualize training

3 Visualizing your model graphs

## Setup Tensorboard

Launch ‘Terminal’ Apps in OpenOnDemand :

Within the Terminal Apps, change your working direcotry to your scratch directory:
```
cd /scratch1/$(whoami)
```
```
cd week5/TAC450-DataScience-Fall2025/week5
```


Activate your Conda environment: 
```
conda activate torch-env
```

Launch Tensorboard in your Conda environment: 
```
tensorboard --logdir=runs
```

Right Click on the http://localhost:6006/ link and select ‘Open link’, this will launch a browser and start Tensorboard




