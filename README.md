# VAEDecon
Gene expression deconvolution by single cell generative model

# Installation
```bash
# conda is recommended
conda create -n vaedecon python=3.11
conda activate vaedecon

# optional: setting HDF5 library path for M1 Mac
# refer to: https://stackoverflow.com/a/73030329/2803344
export HDF5_DIR=/opt/homebrew/opt/hdf5 
export BLOSC_DIR=/opt/homebrew/opt/c-blosc

# if you have a GPU, install pytorch with CUDA support first (optional)
# For linux or Windows, please refer to https://pytorch.org/get-started/previous-versions/
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# For Mac OS
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# sometimes you may need to install the package annoy manually on linux
conda install conda-forge::python-annoy
pip install vaedecon
```
