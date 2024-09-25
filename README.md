# QFAE (Q-Function guided Action Exploration)
This repository contains the source code for "QFAE: Q-Function guided Action Exploration for offline deep reinforcement learning"

## Getting started
QFAE is evaluated on MuJoCo continuous control tasks in OpenAI gym. It is trained using PyTorch 1.13.1+cu117 and Python 3.9.
```bash
# install pytorch and other lib
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
## Method
We have found through partial theoretical analysis that policy improvement can be achieved by adding high-return perturbations. 

## Dataset
Our experiment is based on [D4RL](https://github.com/berkeley-rll/d4rl)

## Running the code

```
python run_experiment.py --configs  configs/halfcheetah-medium-v2.yaml
```

## Result

![image](https://github.com/ML-Group-SDU/QFAE/img/result.jpg)
