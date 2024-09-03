
![OpenDPD](https://github.com/lab-emi/OpenDPD/assets/90694322/85aeba7c-a9f3-423d-b4ed-9b8efed09b33)



**OpenDPD** is an end-to-end learning framework built in PyTorch for modeling power amplifiers (PA) and digital pre-distortion. You are cordially invited to contribute to this project by providing your own backbone neural networks, pre-trained models, or measured PA datasets.

This repo mainly contains the training code of OpenDPD using the baseband signal from a digital transmitter.

# Authors & Citation
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2401.08318)
Yizhuo Wu, Gagan Singh, Mohammad Reza Beikmirza, Leo de Vreede, Morteza Alavi, Chang Gao

Department of Microelectronics, Delft University of Technology, 2628 CD Delft, The Netherlands 

If you find this repository helpful, please cite our work.
- [ISCAS 2024] [OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework for Wideband Power Amplifier Modeling and Digital Pre-Distortion](https://ieeexplore.ieee.org/abstract/document/10558162)
```
@article{Wu2024ISCAS,
  title={Open{DPD}: {A}n {O}pen-{S}ource {E}nd-to-{E}nd {L}earning \& {B}enchmarking {F}ramework for {W}ideband {P}ower {A}mplifier {M}odeling and {D}igital {P}re-{D}istortion},
  author={Wu, Yizhuo and Singh, Gagan Deep and Beikmirza, Mohammadreza and C. N. de Vreede, Leo and Alavi, Morteza and Gao, Chang},
  journal={arXiv preprint arXiv:2401.08318},
  year={2024}
}
```
- [IMS/MWTL 2024] [MP-DPD: Low-Complexity Mixed-Precision Neural Networks for Energy-Efficient Digital Pre-distortion of Wideband Power Amplifiers](https://ieeexplore.ieee.org/document/10502240)
```
@ARTICLE{Wu2024IMS,
  author={Wu, Yizhuo and Li, Ang and Beikmirza, Mohammadreza and Singh, Gagan Deep and Chen, Qinyu and de Vreede, Leo C. N. and Alavi, Morteza and Gao, Chang},
  journal={IEEE Microwave and Wireless Technology Letters}, 
  title={MP-DPD: Low-Complexity Mixed-Precision Neural Networks for Energy-Efficient Digital Predistortion of Wideband Power Amplifiers}, 
  year={2024},
  volume={},
  number={},
  pages={1-4},
  keywords={Deep neural network (DNN);digital predistortion (DPD);digital transmitter (DTX);power amplifier (PA);quantization},
  doi={10.1109/LMWT.2024.3386330}}
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
In this section, we introduce the methods of E2E learning architecture and the **corresponding command line** for them.

<img style="float: left" src="OpenDPD.png" alt="drawing"/> 

The End-to-End (E2E) learning framework encompasses three principal components, as outlined:

**1.Data Acquisition & Pre-Processing:** This phase involves the collection and preprocessing of baseband I/Q signals from the Power Amplifier (PA). To mitigate the issue of gradient vanishing and to augment the efficacy of the training process, the raw data is segmented into shorter frames. The dataset dictionary includes three distinct signal bandwidths from a digital transmitter. The dataset is partitioned in a ratio of 8:2:2 for training, testing, and validation, respectively, to facilitate the learning process.

**2.PA Modeling:** The process involves training a behavioral model of the PA using the framed input and target output through a sequence-to-sequence learning approach, employing Backpropagation Through Time (BPTT) for optimization. 

Command line for step 2 (Change accelerator to cuda if you want to use NVIDIA GPU acceleration):
```
python main.py --dataset_name DPA_200MHz --step train_pa --accelerator cpu
```

**3.DPD Learning:** Subsequently, a Digital Pre-Distortion (DPD) model is integrated prior to the pre-trained PA behavioral model. The input signal is fed to the input of the cascaded model, and through the application of BPTT, the goal is to align the output signal with the amplified linear input signal.

Command line for step 3:
```
python main.py --dataset_name DPA_200MHz --step train_dpd --accelerator cpu
```
***4.Validation experiment:** To assess the DPD model's performance on another PA, it is necessary to generate an ideal input signal post the training of the aforementioned phases. The resultant signal is denominated according to the DPD model specifications and archived in a .csv format within a run_dpd file.

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


## MP-DPD

Additionally, the manuscript introduces MP-DPD, a technique designed to train a fixed-point quantized DPD model without significantly compromising accuracy, ensuring efficient hardware implementation. To reproduce the results in MP-DPD, following command line for each step can be used: 

1. **Pretrain a DPD Model**:

```bash
python main.py --dataset_name DPA_200MHz --step train_dpd --accelerator cpu --DPD_backbone qgru --quant --q_pretrain True
```

2. **Quantization Aware Training of the Pretrained Model**:

```bash
# 16-bit Quantization
# Replace ${pretrained_model_from_previous_step} with the path to the pretrained model
# Replace ${label_for_quantized_model} with a label for the quantized model
python main.py --dataset_name DPA_200MHz --step train_dpd --accelerator cpu --DPD_backbone qgru --quant --n_bits_w 16 --n_bits_a 16 --pretrained_model ${pretrained_model_from_previous_step} --quant_dir_label ${label_for_quantized_model}
```

3. **Output of the Quantized DPD Model**:

```bash
# Make sure the ${label_for_quantized_model} is the same as the one in Step 2
python main.py --dataset_name DPA_200MHz --step run_dpd --accelerator cpu --DPD_backbone qgru --quant --n_bits_w 16 --n_bits_a 16 --quant_dir_label ${label_for_quantized_model}
```

Also, you can reproduce the MP-DPD results with this script:

```bash
bash quant_mp_dpd.sh
```
