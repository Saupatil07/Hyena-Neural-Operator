# Hyena Neural Operator
<!-- <p align="center" width="100%" height="100%">
    <img width="50%" height="50%" src="./images/hyena lowpoly(1).jpg">
</p> -->
<p align="center" width="100%" height="100%">
<img src="./images/hyena lowpoly(1).jpg" width="500px"></img>
</p>

## Datasets for 1D Diffusion-Reaction/2D Navier-Stokes

The dataset for 1D Diffusion-Reaction can be downloaded from [dataset link](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) .</br>
We provide our processed dataset for 2D Navier-Stokes (in .npy format) at [dataset link](https://drive.google.com/drive/folders/1z-0V6NSl2STzrSA6QkzYWOGHSTgiOSYq?usp=sharing) .</br>
The dataset for these problems are under the courtesy of [FNO](https://github.com/zongyi-li/fourier_neural_operator).

### Pretrained model checkpoint

| Problem       | link   |
|---------------|---------------------------------------------------------------------------|
| Navier-Stokes  |  [link](https://drive.google.com/drive/folders/1o_j_4ilbfHHftGmM3_P1UEL_VPPLHTKd?usp=drive_link) |
| Diffusion-Reaction   |  [link](https://drive.google.com/drive/folders/1qGc3deWZ1SKbfwsBk5rxZtuTq8_17hzb?usp=sharing) |

## Usage
* Train on Navier-Stokes dataset:

```bash
python tune_navier_stokes.py \
--lr 1e-4 \
--ckpt_every 10000 \
--iters 96000 \
--batch_size 4 \
--in_seq_len 10 \
--out_seq_len 20 \
--dataset_path ../pde_data/fno_ns_Re200_N10000_T30.npy \   # path to the dataset
--in_channels 12 \
--out_channels 1 \
--encoder_emb_dim 96 \
--out_seq_emb_dim 192 \
--encoder_depth 5 \
--decoder_emb_dim 384 \
--propagator_depth 1 \
--out_step 1 \
--train_seq_num 9800 \
--test_seq_num 200 \
--fourier_frequency 8 \
--encoder_heads 1 \
--use_grad \
--curriculum_ratio 0.16 \
--curriculum_steps 10 \
--aug_ratio 0.0
```
 * Train on 1D Diffusion-Reaction: 

```bash
python hyena_diff_1024.py --iters 90000 --batch_size 20 --ckpt_every 5000 --resolution 1024 --device cuda:1
```

## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/HazyResearch/safari

https://github.com/neuraloperator/neuraloperator/tree/master

https://github.com/pdebench/PDEBench
