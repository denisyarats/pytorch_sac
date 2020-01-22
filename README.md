# Soft Actor-Critic (SAC) implementation in PyTorch

This is PyTorch implementation of Soft Actor-Critic (SAC) [[ArXiv]](https://arxiv.org/abs/1812.05905).

## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment by running:
```
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with:
```
source activate pytorch_sac
```

## Instructions
To train an SAC agent on the `cheetah run` task run:
```
python train.py env=cheetah_run
```
This will produce 'exp' folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir exp
```
and opening up tensorboad in your browser.

The console output is also available in a form:
```
| train | E: 35 | S: 35000 | R: 28.4543 | D: 25.7 s | BR: 0.0341 | ALOSS: -7.8298 | CLOSS: 0.0115 | TLOSS: 0.0227 | TVAL: 0.0076 | AENT: -3.0143
```
a training entry decodes as:
```
train - training episode
E - total number of episodes 
S - total number of environment steps
R - episode reward
D - duration in seconds to train 1 episode
BR - average reward of a sampled batch
ALOSS - average loss of actor
CLOSS - average loss of critic
TLOSS - average loss of temperature
TVAL - average value of temperature
AENT - average entropy of actor
```
while an evaluation entry:
```
| eval | S: 0 | R: 21.1676
```
which just tells the expected reward `R` evaluating current policy after `S` steps. Note that `R` is average evaluation performance over `num_eval_episodes` episodes (usually 10).

## Results
An extensive benchmarking of SAC on the DM Control Suite against D4PG. We plot an average and performance of SAC together with p95 confidence intervals. Note that results for D4PG are reported after 10^8 steps and taken from the original paper.
![Results](figures/dm_control.png)
