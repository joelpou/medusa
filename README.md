# medusa trading bot

A simple trading bot for stocks that infers if trader should buy, hold or sell built with PyTorch. Trained CNN with PyTorch using K-Fold Cross Validation.

## Installation
1. Install python v3.9+
Use the package manager [pip3](https://pip.pypa.io/en/stable/) to install dependencies.

2. Install [Anaconda](https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da) and create new conda virtual environment

```bash
conda create --name medusa-env python=3.10 # create new conda environment
```

3. Install Python requirements: [Pytorch](https://pytorch.org/), pandas and scikit-learn, matplotlib

```bash
nvidia-smi # check your CUDA version before installing pytorch

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

conda install -c pandas

conda install -c anaconda scikit-learn

conda install -c conda-forge matplotlib

conda install -c conda-forge yfinance

conda install -c conda-forge mplfinance

conda install -c conda-forge ta
```

## Usage

```bash
# train CNN
# make sure structure of dataset path contains DOWN, SIDE and UP folders with labeled images
python3 test/train_cnn.py <DATASET_PATH> 

# results of CNN
python3 test/results_cnn.py <PKL_FILE_PATH>

# test inference
python3 test/inference.py <IMAGE_PATH> # TODO input batch of images

# generate random dataset from S&P500
python3 test/get_stock_image_data.py <OUTPUT_DIR_PATH> # TODO input batch of images and display results
```
