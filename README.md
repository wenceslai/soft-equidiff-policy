# New Machine Setup:

---

# 1. System deps (FFmpeg for video decoding, gfortran for escnn, build tools for lie_learn)
sudo apt-get update
sudo apt-get install -y ffmpeg gfortran build-essential

# 2. Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 3. Python 3.12 venv (lerobot requires >=3.12)
uv python install 3.12
uv venv --python 3.12 ~/.venv
source ~/.venv/bin/activate

# 3.5 Setup ssh keys in github

# 4. Clone repos
git clone git@github.com:huggingface/lerobot.git
git clone git@github.com:wenceslai/soft-equidiff-policy.git
cd soft-equidiff-policy

# 5. Install PyTorch (CUDA 12.8 — adjust cu version to match your driver)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 6. Install lerobot from local clone
uv pip install -e ../lerobot/

# 7. Install Cython first (needed to build lie_learn)
uv pip install Cython

# 8. Install lie_learn from source, compiled against current numpy
uv pip install git+https://github.com/AMLab-Amsterdam/lie_learn --no-build-isolation

# 9. Fix missing lie_learn data file
curl -L "https://github.com/AMLab-Amsterdam/lie_learn/raw/master/lie_learn/representations/SO3/pinchon_hoggan/J_dense_0-150.npy" \
  -o ~/.venv/lib/python3.12/site-packages/lie_learn/representations/SO3/pinchon_hoggan/J_dense_0-150.npy

# 10. Install remaining deps
uv pip install escnn einops diffusers wandb matplotlib

# 11. Install this project
uv pip install -e .

# 12. (Optional) For success rate eval
uv pip install gym_pusht gymnasium
uv pip install "pymunk<7"

# 13. Verify installation
python -c "import torch; import escnn; import lerobot; import wandb; print('All good')"

# 14. 
export WANDB_API_KEY="..."


tmux new -s train

python -m soft_equidiff.train \
    --run_name baseline_no_softening \
    --wandb_project soft-equidiff-policy \
    --no_soften_image \
    --no_soften_state \
    --no_soften_action \
    --no_soften_decoder

# to exist control + b, d

tmux attach -t train


python -m soft_equidiff.eval_success_rate \
    --checkpoint outputs/baseline_no_softening/policy_step0050000.pt \
    --n_episodes 50 \
    --device cuda \
    --wandb_project soft-equidiff-policy \
    --wandb_run_name baseline_no_softening_eval