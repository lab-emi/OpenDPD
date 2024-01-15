
![ODPD](https://github.com/lab-emi/OpenDPD/assets/90694322/7a44fbfd-b12c-413e-b50f-473bb17990b0)


**OpenDPD** is an end-to-end learning framework built in PyTorch for power amplifier (PA) modeling and digital pre-distortion. You are cordially invited to contribute to this project by providing your own backbone neural networks, pretrained models or measured PA datasets.

This repo mainly contains the training code of OpenDPD using the baseband signal from a digital transmitter.

# Authors & Citation

Yizhuo Wu, Gagan Singh, Mohammad Reza Beikmirza, Leo de Vreede, Morteza Alavi, Chang Gao

Department of Microelectronics, Delft University of Technology, 2628 CD Delft, The Netherlands 

If you find this repository helpful, please cite our work.

* [ISCAS 2024] OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework for Wideband Power Amplifier Modeling and Digital Pre-Distortion
# Introduction
<img style="float: left" src="OpenDPD.png" alt="drawing" width="200"/> The Figure shows the 

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
## Environment

## Reproduction
Example uses RFWeblab data, here args.gain = 290.

For Matlab data, args.gain = 23.

if other PA are used, please set --gain **mean(Output_amplitude/Input_amplitude)**.

Here, the gain only reflects the amplitude gain of data sets.

### 1. OFDM generator
```
python main.py --dataset_name RFWeblab --step ofdm_gen
```

### 2. train PA
```
python main.py --dataset_name RFWeblab --step train_pa --phase **Your_file_Name**
```

### 3.train DPD
if the second step isn't skipped:
```
python main.py --dataset_name RFWeblab --step train_dpd --phase **Your_file_Name**
```
else we have a prepared RFWeblab PA model:
```
python main.py --dataset_name RFWeblab --step train_dpd --phase **Your_file_Name** --PA_model_phase **Your_file_Name**
```

### 4.Generate ideal input and check simulated ACPR

```
python main.py --dataset_name RFWeblab --step ideal_input_gen --phase **Your_file_Name** --PA_model_phase **Your_file_Name**
```

While checking simulated ACPR of RNN-based model,please set the length of testIn as fft length.

While checking simulated ACPR of GMP, please set the length of testIn as fftlength+frame_length

While checking simulated ACPR of CNN-based model, please change the length of paoutput_len and use the whole testIn data.


