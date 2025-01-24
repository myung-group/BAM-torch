This is a version under development.


Please check the file below for the next steps.
```bam_torch/training/mve_trainer.py```


Installation

```
$ conda create --name bam_torch python=3.11
$ conda activate bam_torch
$ pip install numpy scipy matscipy torch
$ python -c "import torch; print(torch.__version__)"
>>> 2.5.1+cu124
$ python -c "import torch; print(torch.version.cuda)"
>>> 12.4
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
For example,
```
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```
And then,
```
$ pip install pandas mendeleev pytorch_warmup
```
Finally, install BAM-torch
```
$ pip install -e .
```
