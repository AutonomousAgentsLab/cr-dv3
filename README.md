# Curious Replay for Model-based Adaptation

Implementation of [Curious Replay](https://arxiv.org/abs/2306.15934) with the [DreamerV3](https://danijar.com/dreamerv3) agent in [jax](https://github.com/google/jax). 

![fig_overview-01_small](https://user-images.githubusercontent.com/903830/236350331-b7aacb2c-671a-4137-90c2-b4dd210ebf30.png)

If you find this code useful, please reference in your paper:

```
@article{kauvar2023curious,
  title={Curious Replay for Model-Based Adaptation},
  author={Kauvar, Isaac and Doyle, Chris and Zhou, Linqi and Haber, Nick},
  journal={International Conference on Machine Learning},
  year={2023}
}
```


## Method

Curious Replay prioritizes sampling of past experiences for training the agent's world model, 
by focusing on the experiences that are most interesting to the agent - 
whether because they are unfamiliar or surprising. 
Inspired by the concept of curiosity, which is often used as an intrinsic reward to guide action selection, 
here curiosity signals are used to guide selection of what experiences the agent should learn from (i.e. train its model with). 

Curious Replay is a simple modification to existing agents that use experience replay -- with minimal 
computational overhead -- by leveraging a count of how many times an experience has been sampled 
and the model losses that are computed for each training batch. 

This prioritization is especially helpful in changing environments, where adaptation is necessary. 
Curious Replay helps keep the world model up to date as the environment changes, which is
a prerequisite for effective action selection in model-based architectures. 

# Install instructions on a fresh Ubuntu 22.04 (x86) install
```bash
sudo apt install build-essential -y

# Replace ubuntu2204 with your ubuntu version if it's different
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb

sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-11-8 -y

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc


# reload the bashrc to set the cuda path
source ~/.bashrc

sudo apt-get install libcudnn8=8.8.0.121-1+cuda11.8
sudo apt-get install libcudnn8-dev=8.8.0.121-1+cuda11.8

mkdir src
cd src
# (optional) git config --global credential.helper store
git clone https://github.com/AutonomousAgentsLab/curiousreplay-dv3.git

cd curiousreplay-dv3
git checkout release-working-ik

sudo apt install python-is-python3 python3.10-venv ffmpeg -y

# Create and activate a virtual environment
python -m venv ~/src/envs/dv3
source ~/src/envs/dv3/bin/activate

pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
pip install dm-reverb
```

# Run Curious Replay
```bash
# Run curious-replay crafter
python dreamerv3/train.py --logdir ~/logdir/crafter-dv3-cr_1 \
--env.crafter.outdir ~/logdir/crafter-dv3-cr_1 --configs crafter --replay curious-replay

# Run curious-replay DMC
python dreamerv3/train.py --logdir ~/logdir/dmc_vision-dv3-cr_1 \
--configs dmc_vision --replay curious-replay --envs.amount 1 --task dmc_walker_walk

# Run baseline crafter
python dreamerv3/train.py --logdir ~/logdir/crafter-dv3_1 \
--env.crafter.outdir ~/logdir/crafter-dv3_1 --configs crafter

# Run tensorboard
tensorboard --logdir ~/logdir/crafter-dv3-cr_1 

# Summarize crafter results
pip install pandas matplotlib
python dreamerv3/plot_crafter.py
```

# Limitations

* No support for parallel environments, so it may need to be run with `--envs.amount 1` flag to override the default number of envs. 
* No support for resuming runs
* These setup instructions have only been tested on Google Cloud Platform Ubuntu 22.04 (x86) with A100 (40 GB).
* Based on fork of [DreamerV3 commit 84ecf19](https://github.com/danijar/dreamerv3/tree/84ecf191d967f787f5cc36298e69974854b0df9c).

