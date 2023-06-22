Fork of https://github.com/danijar/dreamerv3 on February 27, 2023

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

