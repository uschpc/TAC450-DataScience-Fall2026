# Setting up a Nvidia Rapids (CUDA-X for Data Science) with Jupyter kernel on a GPU node
Here are the steps to Install Nvidia Rapids (or CUDA-X for Data Science on CARC)

## 1. Get an interactive GPU node

Build the environment on a GPU node, not the login node.

```bash
salloc -p gpu -N 1 -n 10 --gres=gpu:1 --time=1:00:00 --account=irahbari_1147
```

Wait for the job to be allocated; you should then be on the compute node. Run `nvidia-smi` to confirm a GPU is listed.

## 2. Create the conda environment

```bash
conda create -n rapids-26.08 -c rapidsai -c conda-forge \
    cudf=26.08 python=3.12 'cuda-version>=12.2,<=12.9'

conda activate rapids-26.08
```
This installs basic packages that come with Nvidia-Rapids as well as CuDF (Pandas alternative for GPUs). The `cuda-version` range keeps the CUDA libraries on CUDA 12, which the node's driver must support (`nvidia-smi` shows the driver's CUDA version in its header). To find all the available packages, installation command for alternative cuda versions and more, visit: https://docs.nvidia.com/datascience/install/#install-the-libraries

## 3. Register the Jupyter kernel

```bash
mamba install -c rapidsai -c conda-forge matplotlib ipykernel -y
python -m ipykernel install --user --name rapids-26.08 --display-name rapids-26.08
```

`ipykernel` lets Jupyter run code in this environment, and `--user` writes the kernel spec to `~/.local/share/jupyter/kernels/rapids-26.08`. Any Jupyter session you start on the cluster will then list **rapids-26.08** as a kernel. Start that session on a GPU node, or cuDF has no GPU to run on. `conda install` works in place of `mamba`; mamba is just faster.

## 4. Test it

Open a notebook with the rapids-26.08 kernel and run:

```python
import os
os.environ["CUDA_PATH"] = os.path.expanduser("~/.conda/envs/rapids-26.08/targets/x86_64-linux")

import numpy as np
import cupy as cp
import cudf as cf

print(f'cupy version: {cp.__version__}')
print(f'numpy version: {np.__version__}')
print(f'CuDF version: {cf.__version__}')
print(cf.Series([1, 2, 3]).sum())  # a small computation on the GPU
```

`CUDA_PATH` points CuPy at the CUDA toolkit files inside the environment, which it needs when it compiles GPU kernels at run time, so set it before importing CuPy.

If the three versions print and the last line prints `6`, cuDF is running on the GPU.
