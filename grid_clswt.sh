#!/bin/bash
set -x

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export MASTER_PORT=${MASTER_PORT:-12320}
export OMP_NUM_THREADS=1

# Base directories
OUTPUT_DIR='/home/hpc12/Lubaina/FocusMAE/Dataset_folder/work_dir/finetune_output/checkpoint_folder'
DATA_PATH='/home/hpc12/Lubaina/FocusMAE/Dataset_folder/BUV'
MODEL_PATH='/home/hpc12/Lubaina/FocusMAE/Dataset_folder/work_dir/pretrain_output/checkpoint_folder/pre.pt'

# Create NEW results directory
RESULTS_DIR="${OUTPUT_DIR}_learning_rate_search_$(date +%Y%m%d_%H%M)"
mkdir -p ${RESULTS_DIR}

# Create a proper CSV file with headers
echo "Run,Learning Rate,Val Accuracy,Test Accuracy,Sensitivity,Specificity,F1 Score,Balanced Score" > ${RESULTS_DIR}/results.csv

# Define learning rate values to test
LEARNING_RATES=("1e-4" "5e-5" "7e-5" "1e-6")

# Track best performance
BEST_BALANCED=0
BEST_RUN=0
BEST_LR=""

# Run counter
run_id=0

# Loop through learning rates
for lr in "${LEARNING_RATES[@]}"; do
    # Increment run counter
    run_id=$((run_id+1))
    
    echo "======= Run $run_id: Learning Rate=$lr ======="
    
    # Clear GPU memory
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # Create run directory with descriptive name
    RUN_DIR="${RESULTS_DIR}/run_${run_id}_lr${lr}"
    mkdir -p $RUN_DIR
    LOG_FILE="${RUN_DIR}/train.log"

    # Run training with current learning rate
    torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=1 \
    ../run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set GBC_Net \
        --nb_classes 2 \
        --no_auto_resume \
        --save_ckpt \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${RUN_DIR} \
        --output_dir ${RUN_DIR} \
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 2 \
        --opt adamw \
        --num_segment 1 \
        --lr ${lr} \
        --class_weights 6 1 \
        --min_lr 1e-8 \
        --warmup_epochs 10 \
        --opt_betas 0.9 0.95 \
        --test_num_segment 20 \
        --test_num_crop 7 \
        --epochs 15 \
        --test_randomization \
        --dist_eval \
        --enable_deepspeed \
        --weight_decay 0.02 \
        --layer_decay 0.9 \
        --update_freq 8 \
        --drop_path 0.15 \
        --drop 0.1 \
        --aa rand-m7-mstd0.5-inc1 \
        --with_checkpoint \
        2>&1 | tee ${LOG_FILE}
    
    # Extract metrics from log file
    metrics=$(extract_metrics ${LOG_FILE})
    IFS=',' read -ra metrics_array <<< "$metrics"
    val_acc="${metrics_array[0]}"
    test_acc="${metrics_array[1]}"
    sensitivity="${metrics_array[2]}"
    specificity="${metrics_array[3]}"
    f1_score="${metrics_array[4]}"
    
    # Calculate balanced score (average of sensitivity and specificity)
    balanced_score=$(echo "scale=4; ($sensitivity + $specificity) / 2" | bc)
    
    # Save results to CSV
    echo "$run_id,$lr,$val_acc,$test_acc,$sensitivity,$specificity,$f1_score,$balanced_score" >> ${RESULTS_DIR}/results.csv

    # Track the best learning rate
    if (( $(echo "$balanced_score > $BEST_BALANCED" | bc -l) )); then
        BEST_BALANCED=$balanced_score
        BEST_RUN=$run_id
        BEST_LR=$lr
    fi

    echo "Finished run $run_id: Learning Rate=$lr"
    echo "Current best: Run $BEST_RUN (LR=$BEST_LR) with balanced score $BEST_BALANCED"
    
    sleep 5
done

# Print final results
echo "============== GRID SEARCH COMPLETE ================"
echo "Best performance: Run $BEST_RUN with learning rate $BEST_LR"
echo "Balanced score (avg of sensitivity & specificity): $BEST_BALANCED"
echo "Full results available in: ${RESULTS_DIR}/results.csv"
