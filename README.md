# spiky lib, version 0.8
Effective CUDA implementation of several spiking neural network models

# Installation

git clone git@github.com:anatoli-starostin/spiky.git

cd spiky

python3.11 -m venv ./.venv --system-site-packages

. ./.venv/bin/activate

pip install -r requirements.txt

pip install -e .

cd ./spiky_cuda

pip install -e .

cd ../