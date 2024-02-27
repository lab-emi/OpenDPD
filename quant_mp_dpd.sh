#!/bin/bash

# Arguments
while getopts g: option
do
case "${option}"
in
g) gpu_device=${OPTARG};;
esac
done

# Global Settings
dataset_name=DPA_200MHz
accelerator=cpu
devices=0

# Hyperparameters
seed=0
n_epochs=100
frame_length=50
frame_stride=1
loss_type=l2
opt_type=adamw
batch_size=64
batch_size_eval=256
lr_schedule=1
lr=1e-3
lr_end=1e-6
decay_factor=0.5
patience=10

#########################
# Train
#########################
seed=(0)

# PA Model
PA_backbone=dgru
PA_hidden_size=8
PA_num_layers=1

# DPD Model
DPD_backbone=(qgru)
DPD_hidden_size=(10)
DPD_num_layers=(1)

# Quantization
quant_n_bits_w=16
quant_n_bits_a=16
quant_dir_label='w'${quant_n_bits_w}'a'${quant_n_bits_a}
quant_opts='--quant'
q_pretrain='True'

function echo_green {
    tput setaf 2
    echo $1
    tput sgr0
}

# Train a PA model
echo_green "==== Train PA model ===="
python main.py --dataset_name ${dataset_name} --step train_pa --accelerator ${accelerator} --n_epochs ${n_epochs} || exit 1;

for i_seed in "${seed[@]}"; do
    for ((i=0; i<${#DPD_backbone[@]}; i++)); do
        # pre-train DPD
        # echo with green color said the pre-train is running
        echo_green "==== Pre-training DPD with backbone ${DPD_backbone[$i]} and hidden size ${DPD_hidden_size[$i]} ====" 
        step=train_dpd
        q_pretrain='True'
        quant_dir_label=''
        pretrained_model=''
        python main.py --dataset_name "$dataset_name" --seed "$i_seed" --step "$step"\
        --accelerator "$accelerator" --devices "$devices"\
        --PA_backbone "$PA_backbone" --PA_hidden_size "$PA_hidden_size" --PA_num_layers "$PA_num_layers"\
        --DPD_backbone "${DPD_backbone[$i]}" --DPD_hidden_size "${DPD_hidden_size[$i]}" --DPD_num_layers "${DPD_num_layers[$i]}"\
        --frame_length "$frame_length" --frame_stride "$frame_stride" --loss_type "$loss_type" --opt_type "$opt_type"\
        --batch_size "$batch_size" --batch_size_eval "$batch_size_eval" --n_epochs "$n_epochs" --lr_schedule "$lr_schedule"\
        --lr "$lr" --lr_end "$lr_end" --decay_factor "$decay_factor" --patience "$patience" \
        "$quant_opts" --n_bits_w "$quant_n_bits_w" --n_bits_a "$quant_n_bits_a" --pretrained_model "$pretrained_model" \
        --quant_dir_label "$quant_dir_label" --q_pretrain "$q_pretrain" \
        || exit 1;

        # quantized aware training DPD
        step=train_dpd
        echo_green "==== Quantized aware training DPD with backbone ${DPD_backbone[$i]}: hidden_size ${DPD_hidden_size[$i]}; quantization: w${quant_n_bits_w}a${quant_n_bits_a} ===="
        q_pretrain=''
        quant_dir_label='w'${quant_n_bits_w}'a'${quant_n_bits_a}

    # please manually change the path of the pre-trained model if the DPD model size is changed
    pretrained_model='./save/'${dataset_name}'/train_dpd/DPD_S_'${i_seed[$i]}'_M_QGRU_H_'${DPD_hidden_size[$i]}'_F_'${frame_length}'_P_502.pt'
        python main.py --dataset_name "$dataset_name" --seed "$i_seed" --step "$step"\
        --accelerator "$accelerator" --devices "$devices"\
        --PA_backbone "$PA_backbone" --PA_hidden_size "$PA_hidden_size" --PA_num_layers "$PA_num_layers"\
        --DPD_backbone "${DPD_backbone[$i]}" --DPD_hidden_size "${DPD_hidden_size[$i]}" --DPD_num_layers "${DPD_num_layers[$i]}"\
        --frame_length "$frame_length" --frame_stride "$frame_stride" --loss_type "$loss_type" --opt_type "$opt_type"\
        --batch_size "$batch_size" --batch_size_eval "$batch_size_eval" --n_epochs "$n_epochs" --lr_schedule "$lr_schedule"\
        --lr "$lr" --lr_end "$lr_end" --decay_factor "$decay_factor" --patience "$patience" \
        "$quant_opts" --n_bits_w "$quant_n_bits_w" --n_bits_a "$quant_n_bits_a" --pretrained_model "$pretrained_model" \
        --quant_dir_label "$quant_dir_label" --q_pretrain "$q_pretrain" \

        # Run DPD
        echo_green "==== Run DPD with backbone ${DPD_backbone[$i]}: hidden_size ${DPD_hidden_size[$i]}; quantization: w${quant_n_bits_w}a${quant_n_bits_a} ===="
        step=run_dpd
        python main.py --dataset_name "$dataset_name" --seed "$i_seed" --step "$step"\
          --accelerator "$accelerator" --devices "$devices"\
          --PA_backbone "$PA_backbone" --PA_hidden_size "$PA_hidden_size" --PA_num_layers "$PA_num_layers"\
          --DPD_backbone "${DPD_backbone[$i]}" --DPD_hidden_size "${DPD_hidden_size[$i]}" --DPD_num_layers "${DPD_num_layers[$i]}"\
          --frame_length "$frame_length" --frame_stride "$frame_stride" --loss_type "$loss_type" --opt_type "$opt_type"\
          --batch_size "$batch_size" --batch_size_eval "$batch_size_eval" --n_epochs "$n_epochs" --lr_schedule "$lr_schedule"\
          --lr "$lr" --lr_end "$lr_end" --decay_factor "$decay_factor" --patience "$patience" \
          "$quant_opts" --n_bits_w "$quant_n_bits_w" --n_bits_a "$quant_n_bits_a" --pretrained_model "$pretrained_model"  \
          --quant_dir_label "$quant_dir_label" \
          || exit 1;
    done
done
