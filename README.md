# Yizhuo_DPD
Now GMP, GRU, LSTM, VDLSTM, PGJANET, DVRJANET, 1DCNN, 2DCNN(RVTDCNN), RGRU Model are avaliable.

Input Features for GRU, LSTM: I and Q

Input Features for RVTDCNN: I, Q, |x|, |x|^2, |x|^3, sin\theta cos\theta

Input Features for RGRU: I, Q, |x|, |x|^3, sin\theta cos\theta

## Settings for Different NN model
GMP: frame_length = memory_length;  degree = degree

1DCNN: hidden_size = kernel_size

2DCNN(RVTDCNN): hidden_size=kernel_number; NN_H=kernel_height; NN_W=kernel_width; cnn_memory=input_matrix_memory_length;
                paoutput_len = output_length_of_pa_model;

## Run
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


