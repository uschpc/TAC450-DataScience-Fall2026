# TAC week5 

Tensorboard Tutorial Link: https://pytorch.org/tutorials/beginner/introyt/tensorboardyt_tutorial.html


## Software Environment Setup

First login to CARC OnDemand: https://ondemand.carc.usc.edu/ and request a 'Discovery Cluster Shell Access' within OpenOnDemand. 

Change your working direcotry to your scratch directory and clone the repo:
```
cd /scratch1/$(whoami)
```
```
mkdir week5
cd week5
```
```
git clone https://github.com/uschpc/TAC450-DataScience-Fall2026
cd TAC450-DataScience-Fall2026/week5
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
cd week5/TAC450-DataScience-Fall2026/week5
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


Note: if you have issues launching traveler desktop (when you click 'Launch Traveller Desktop', you see a black screen): 
Please go to home directory, go to .config folder and rename xfce4 folder to something like xfce4-old and relaunch another session. 

OR simply remove the folder
```
rm -rf ~/.config/xfce4
```
