#!/bin/bash
set -x

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Increased from 128
export MASTER_PORT=${MASTER_PORT:-12320}
export OMP_NUM_THREADS=1

# Base directories
OUTPUT_DIR='/home/hpc12/Lubaina/FocusMAE/Dataset_folder/work_dir/finetune_output/checkpoint_folder'
DATA_PATH='/home/hpc12/Lubaina/FocusMAE/Dataset_folder/BUV'
MODEL_PATH='/home/hpc12/Lubaina/FocusMAE/Dataset_folder/work_dir/pretrain_output/checkpoint_folder/checkpoint-149.pth'
RESULTS_DIR="${OUTPUT_DIR}_grid_search_results"

# Create results directory
mkdir -p ${RESULTS_DIR}
echo "Learning Rate,Validation Accuracy,Test Accuracy" > ${RESULTS_DIR}/grid_search_results.csv

# Define learning rate values to test
LEARNING_RATES=("1e-4" "1e-5" "7e--5" "1e-6")

# Common parameters
N_NODES=${N_NODES:-1}
GPUS_PER_NODE=1
NODE_RANK=0

# Function to extract accuracy from log file
extract_accuracy() {
    local log_file=$1
    # Extract the highest validation accuracy and final test accuracy
    local val_acc=$(grep "Accuracy of the network on the validation set:" $log_file | awk '{print $NF}' | sort -nr | head -1)
    local test_acc=$(grep "Accuracy of the network on the test set:" $log_file | tail -1 | awk '{print $NF}')
    echo "$val_acc,$test_acc"
}

# Run for each learning rate
for lr in "${LEARNING_RATES[@]}"; do
    echo "Starting training with learning rate: $lr"
    
    # Clear GPU memory before each run
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # Create run-specific output directory
    RUN_DIR="${OUTPUT_DIR}_Bv_grid_lr_${lr}"
    mkdir -p $RUN_DIR
    LOG_FILE="${RESULTS_DIR}/lr_${lr}.log"
    
    # Run the training with current learning rate
    torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=${GPUS_PER_NODE} \
    ../run_class_finetuning.py \
            --model vit_base_patch16_224 \
            --data_set GBC_Net \
            --nb_classes 2 \
            --save_ckpt \
            --data_path ${DATA_PATH} \
            --finetune ${MODEL_PATH} \
            --auto_resume \
            --log_dir ${RUN_DIR} \
            --output_dir ${RUN_DIR} \
            --batch_size 1 \
            --input_size 224 \
            --short_side_size 224 \
            --num_frames 16 \
            --sampling_rate 2 \
            --num_sample 2 \
            --num_workers 4 \
            --opt adamw \
            --num_segment 1 \
            --lr ${lr} \
            --min_lr 1e-7 \
            --warmup_epochs 5 \
            --opt_betas 0.9 0.999 \
            --test_num_segment 10 \
            --test_num_crop 3 \
            --epochs 15 \
            --test_randomization \
            --dist_eval \
            --enable_deepspeed \
            --weight_decay 0.05 \
            --layer_decay 0.75 \
            --update_freq 1 \
            --drop_path 0.2 \
            --aa rand-m7-mstd0.5-inc1 \
            --mixup 0.8 \
            --cutmix 1.0 \
            --mixup_prob 1.0 \
            --save_ckpt_freq 5 \
            2>&1 | tee ${LOG_FILE}
    
    # Extract and save results
    accuracies=$(extract_accuracy ${LOG_FILE})
    echo "$lr,$accuracies" >> ${RESULTS_DIR}/grid_search_results.csv
    
    echo "Finished training with learning rate: $lr"
    echo "----------------------------------------"
    sleep 10  # Give system time to clean up between runs
done

# Summarize results
echo "Grid search complete. Results saved in ${RESULTS_DIR}/grid_search_results.csv"
echo "Summary of results:"
cat ${RESULTS_DIR}/grid_search_results.csv

# Find best learning rate
BEST_LR=$(sort -t, -k2 -nr ${RESULTS_DIR}/grid_search_results.csv | head -2 | tail -1 | cut -d, -f1)
echo "Best learning rate based on validation accuracy: ${BEST_LR}"
