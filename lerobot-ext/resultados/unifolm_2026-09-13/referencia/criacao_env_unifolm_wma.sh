set -eux
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n unifolm-wma python==3.10.18
conda activate unifolm-wma
# xformers pelo índice cu128: ele arrasta o torch/torchvision casados, que é o
# que traz kernels sm_120. Os pins de torch==2.3.1/xformers==0.0.27 do
# pyproject ficam de fora — eles só têm kernels até sm_90.
pip install xformers --index-url https://download.pytorch.org/whl/cu128
# O resto dos pins do upstream, sem torch/torchvision/xformers e sem o
# tensorflow do dlimp (que só serve para preparar Open-X, não para inferir).
pip install \
  "decord==0.6.0" "einops==0.8.0" "imageio==2.35.1" "imageio-ffmpeg" \
  "omegaconf==2.3.0" "opencv-python==4.10.0.84" "pandas==2.0.0" \
  "pyyaml==6.0.1" "tqdm==4.66.5" "transformers==4.40.1" \
  "av==12.3.0" "timm==0.9.10" "scikit-learn==1.5.1" \
  "open-clip-torch==2.22.0" "kornia==0.7.3" "diffusers==0.30.2" \
  "termcolor==2.4.0" "accelerate==1.7.0" "fairscale==0.4.13" \
  "pytorch-lightning==1.9.3" "h5py" "safetensors" "moviepy==1.0.3"
pip install -e /home/miguel/DEV/prometheus-vla/unifolm-wma --no-deps
