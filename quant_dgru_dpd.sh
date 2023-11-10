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
accelerator=cuda
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
PA_backbone=qgru
PA_hidden_size=8
PA_num_layers=1

# DPD Model
DPD_backbone=(qgru)
DPD_hidden_size=(8 20 11 8 12 8)
DPD_num_layers=(1 2 1 1 1 1)

# Quantization
quant_n_bits_w=8
quant_n_bits_a=8
pretrained_model=''
quant_opts='--quant'

for i_seed in "${seed[@]}"; do
    for ((i=0; i<${#DPD_backbone[@]}; i++)); do
        # Train DPD
        step=train_dpd
        python main.py --dataset_name "$dataset_name" --seed "$i_seed" --step "$step"\
        --accelerator "$accelerator" --devices "$devices"\
        --PA_backbone "$PA_backbone" --PA_hidden_size "$PA_hidden_size" --PA_num_layers "$PA_num_layers"\
        --DPD_backbone "${DPD_backbone[$i]}" --DPD_hidden_size "${DPD_hidden_size[$i]}" --DPD_num_layers "${DPD_num_layers[$i]}"\
        --frame_length "$frame_length" --frame_stride "$frame_stride" --loss_type "$loss_type" --opt_type "$opt_type"\
        --batch_size "$batch_size" --batch_size_eval "$batch_size_eval" --n_epochs "$n_epochs" --lr_schedule "$lr_schedule"\
        --lr "$lr" --lr_end "$lr_end" --decay_factor "$decay_factor" --patience "$patience" \
        "$quant_opts" --n_bits_w "$quant_n_bits_w" --n_bits_a "$quant_n_bits_a" --pretrained_model "$pretrained_model"  \
        || exit 1;
        # Run DPD
        step=run_dpd
        python main.py --dataset_name "$dataset_name" --seed "$i_seed" --step "$step"\
          --accelerator "$accelerator" --devices "$devices"\
          --PA_backbone "$PA_backbone" --PA_hidden_size "$PA_hidden_size" --PA_num_layers "$PA_num_layers"\
          --DPD_backbone "${DPD_backbone[$i]}" --DPD_hidden_size "${DPD_hidden_size[$i]}" --DPD_num_layers "${DPD_num_layers[$i]}"\
          --frame_length "$frame_length" --frame_stride "$frame_stride" --loss_type "$loss_type" --opt_type "$opt_type"\
          --batch_size "$batch_size" --batch_size_eval "$batch_size_eval" --n_epochs "$n_epochs" --lr_schedule "$lr_schedule"\
          --lr "$lr" --lr_end "$lr_end" --decay_factor "$decay_factor" --patience "$patience" \
          "$quant_opts" --n_bits_w "$quant_n_bits_w" --n_bits_a "$quant_n_bits_a" --pretrained_model "$pretrained_model"  \
          || exit 1;
    done
done