# Discretizing SO(2)-Equivariant Features for Robotic Kitting
[PDF](https://arxiv.org/ftp/arxiv/papers/2403/2403.13336.pdf)&nbsp;&nbsp;•&nbsp;&nbsp; **RSS 2022**

*Jiadong Zhou, Yadan Zeng, Huixu Dong, I-Ming Chen*

**Abstract.** Robotic kitting has attracted considerable attention in logistics and industrial settings. However, existing kitting methods encounter challenges such as low precision and poor efficiency, limiting their widespread applications. To address these issues, we present a novel kitting framework that improves both the precision and computational efficiency of complex kitting tasks. Firstly, our approach introduces a fine-grained orientation estimation technique in the picking module, significantly enhancing orientation precision while effectively decoupling computational load from orientation granularity. This approach combines an SO(2)-equivariant network with a group discretization operation to preciously predict discrete orientation distributions. Secondly, we develop the Hand-tool Kitting Dataset (HKD) to evaluate the performance of different solutions in handling orientation-sensitive kitting tasks. This dataset comprises a diverse collection of hand tools and synthetically created kits, which reflects the complexities encountered in real-world kitting scenarios. Finally, a series of experiments are conducted to evaluate the performance of the proposed method. The results demonstrate that our approach offers remarkable precision and enhanced computational efficiency in robotic kitting tasks.

## Hand-tool Kitting Tasks

The kitting simulation is developed based on [Ravens](https://github.com/google-research/ravens) and its [variations](https://github.com/HaojHuang/Equivariant-Transporter-Net), a simulation environment in PyBullet for planar manipulation with emphasis on pick and place. We create a Hand-tool Kitting Dataset (HKD), which focus on the kitting tasks with high orientation sensitivity. It inherits the Gym-like API from Ravens, each with (i) a scripted oracle that provides expert demonstrations and (ii) reward functions that provide partial credit.

[Important] For the dataset, please request access and download from the [website](./todo). Please move the dataset to [raven/assets/](./raven/assets/) for running the tasks.

<img src="image/Manipulation_Sequence.png"><br>

(a) **block-insertion**: pick up the L-shaped red block and place it into the L-shaped fixture.<br>
(b) **place-red-in-green**: pick up the red blocks and place them into the green bowls amidst other objects.<br>
(c) **align-box-corner**: pick vup the randomly sized box and align one of its corners to the L-shaped marker on the tabletop.<br>
(d) **stack-block-pyramid**: sequentially stack 6 blocks into a pyramid of 3-2-1 with rainbow colored ordering.<br>
(e) **palletizing-boxes**: pick up homogeneous fixed-sized boxes and stack them in transposed layers on the pallet.<br>
(f) **assembling-single-toolkit** pick up a seen set of hand tools and place them into desinaated cavities within the kits. <br>
(g) **assembling-kits-3dtoolkit** pick up unseen toolsets and place them into designated cavities within the kits. <br>

## Installation

**Step 1.** Recommended: install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python 3.7.

**Step 2.** Install Pytorch
```commandline
# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
**Step 3.** Install the required packages

```shell
pip install -r ./requirements.txt
pip install kornia
pip install easydict
pip install escnn
pip install trimesh
```

## Getting Started

**Step 1.** Generate training and testing data.
 
```shell
python gripper_get_demo.py  --mode train --task assembling-kits-3dtoolkit --n 100 --disp
python gripper_get_demo.py  --mode test  --task assembling-kits-3dtoolkit --n 100 --disp
```

**Step 2.** Train a model e.g., Equivariant Transporter. Parameters are saved to the `checkpoints_{model}` directory. Note that important condifigurations are saves in `./configs/train.yaml` by default.

```shell
python train.py --task assembling-kits-3dtoolkit --n_demos 100 --n_rotations 36 --agent so2 --n_align 2 --n_steps 10000 --interval 2000 --gpu_id 0 --logging
# --config_file train.yaml --postfix h2 --seed 0
```

**Step 3.** Evaluate the model trained for 200 iterations with 1 demos. Results are saved to the `test_{model}` directory.

```shell
python gripper_test.py --task assembling-kits-3dtoolkit --n_demos 100 --n_rotations 36 --agent so2 --n_align 2 --n_steps 10000 --gpu_id 0 --disp
# --config_file train.yaml --entire --seed 0 --postfix h2
```


**Step 4.** Plot and print results.

```shell
python plot.py --task assembling-kits-3dtoolkit --n_demos 100 --n_rotations 36 --agent so2 --disp
```

**Optional.** Tracking training and validation losses with Tensorboard.
```shell
python -m tensorboard.main --logdir=logs  # Open the browser to where it tells you to.
```

**Observations:** RGB-D images (4X320X160)

**Actions:** Picking: (u,v,theta); Placing: (u,v,theta)


## Robot Commends

**Step 1.** Collect human demonstrations

```
python robot/get_demo.py
```

**Step 2.** Perform validation
```
python robot/robot_val.py
```

**Step 3.** Peform robot testing
```
python robot robot_test.py
```
