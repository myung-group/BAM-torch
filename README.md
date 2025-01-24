## This is a version under development. 


## Please check the file below for the next steps. 
```bam_torch/training/mve_trainer.py```


## Installation 

```
$ conda create --name bam_torch python=3.11
$ conda activate bam_torch
$ pip install numpy scipy matscipy torch

# Check the version of torch and cuda
$ python -c "import torch; print(torch.__version__)"  
>>> 2.5.1+cu124
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
For example,
```
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```
And then,
```
$ pip install pytorch_warmup
```
Finally, install BAM-torch
```
$ pip install -e .
```


## Run
There are examples in ```examples/example-*/```

* For training,
  ```
  python main.py
  ```
* For evaluating,
  ```
  python evaluate.py
  ```
