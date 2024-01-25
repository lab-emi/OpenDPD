
![OpenDPD](https://github.com/lab-emi/OpenDPD/assets/90694322/85aeba7c-a9f3-423d-b4ed-9b8efed09b33)



**OpenDPD** is an end-to-end learning framework built in PyTorch for modeling power amplifiers (PA) and digital pre-distortion. You are cordially invited to contribute to this project by providing your own backbone neural networks, pre-trained models, or measured PA datasets.

This repo mainly contains the training code of OpenDPD using the baseband signal from a digital transmitter.

# Authors & Citation
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2401.08318)
Yizhuo Wu, Gagan Singh, Mohammad Reza Beikmirza, Leo de Vreede, Morteza Alavi, Chang Gao

Department of Microelectronics, Delft University of Technology, 2628 CD Delft, The Netherlands 

If you find this repository helpful, please cite our work.
- [ISCAS 2024, RFIC & AI Special Session] [OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework for Wideband Power Amplifier Modeling and Digital Pre-Distortion](https://arxiv.org/abs/2401.08318)
```
@misc{wu2024opendpd,
      title={OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework for Wideband Power Amplifier Modeling and Digital Pre-Distortion}, 
      author={Yizhuo Wu and Gagan Deep Singh and Mohammadreza Beikmirza and Leo de Vreede and Morteza Alavi and Chang Gao},
      year={2024},
      eprint={2401.08318},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
# Project Structure
```
.
└── backbone        # Configuration Files (for feature extractor or whatever else you like).
└── datasets        # Measured PA Datasets.
    └──DPA_200MHz   └── # Digital Power Amplifier with 200 MHz OFDM Signals
└── dpd_out         # Outputs of  (Automatically generated).
└── log             # Experimental Log Data (Automatically generated).
└── modules         # Major Modules.
└── save            # Saved Models.
└── steps           # Steps (train_pa, train_dpd, run_dpd, all called from main.py).
└── utils           # Libraries of useful methods.
└── argument.py     # Arguments
└── main.py         # Top (Everything starts from here).
└── model.py        # Top-level Neural Network Models
└── project.py      # A class having useful functions and storing hyperparameters

```

# Introduction and Quick Start

## Environment
This project was tested with PyTorch 2.1 in Ubuntu 22.04 LTS.

Install Miniconda (Linux). If you use MacOS, please download [Miniconda3-latest-MacOSX-arm64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh)
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
Create an environment and install the required packages. If you don't use CUDA, please follow the [PyTorch](https://pytorch.org/) official installation instruction.
```
conda create -n pt python=3.11 numpy matplotlib pandas scipy tqdm \
    pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Activate the environment.
```
conda activate pt
```

## End-to-End (E2E) Training
In this section, we introduce the methods of E2E learning architecture and **corresponding command line** for them.

<img style="float: left" src="OpenDPD.png" alt="drawing"/> 

As shown in above Figure, the E2E learning architecture consists of three primary steps:

**1.Data Acquisition & Pre-Processing:** The baseband I/Q signals are sent and sourced from the PA. To address the gradient vanishing issue and enhance training, features and labels are split into shorter frames. In this repo, the datasets dictionary concludes three different bandwidth signals from one digital transmitter. And for the training process, we split the samples according to training:test:validation of 8:2:2 ratio.

**2.PA Modeling:** Using framed input and target output, a PA behavioral model is trained in a sequence-to-sequence learning way via backpropagation Through Time **(BPTT)**. 

Command line for step 2:
```
python main.py --dataset_name DPA_200MHz --step train_pa --accelerator cpu
```

**3.DPD Learning:** A DPD model is cascaded before the pre-trained PA behavioral model. In this configuration, the parameters of the PA model remain unaltered. During step 3, the input signal is fed to the input of the cascaded model. Executing BPTT across a cascaded model, the output aims to converge to the linear amplified input signal.

Command line for step 3:
```
python main.py --dataset_name DPA_200MHz --step train_dpd --accelerator cpu
```
***4.Validation experiment:** If you would like to test the DPD on your own PA, you need to generate the ideal input for PA after training steps 2 and 3. The generated signal is named by its DPD model settings and saved in a run_dpd file in .csv format.

Command line for step 4:
```
python main.py --dataset_name DPA_200MHz --step run_dpd --accelerator cpu
```

## Reproduce the results in OpenDPD

1. Random Seed PA Training: To reproduce the PA modeling results in **OpenDPD** Figure.4(a), the following command line can be used.
```
bash train_all_pa.sh
```
This file will train all kinds of PA model around 500 parameters with 5 random seed. Figure.4(a) shows the average results from these 5 random seed.

2. DPD learning: To reproduce the DPD learning results in **OpenDPD** Figure.4 (b) and Figure.4 (d) and Table 1, the following command line can be used.
```
bash train_all_dpd.sh
```
This file will train all kinds of DPD model around 500 parameters.


