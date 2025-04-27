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

# Create NEW results directory to avoid overwriting previous results
RESULTS_DIR="${OUTPUT_DIR}_improved_params_$(date +%Y%m%d_%H%M)"
mkdir -p ${RESULTS_DIR}

# Create a proper CSV file with headers
echo "Run,Learning Rate,Drop Path,Class Weights,Layer Decay,Val Accuracy,Test Accuracy,Sensitivity,Specificity,F1 Score" > ${RESULTS_DIR}/results.csv


CLASS_WEIGHTS=("0.5 0.5" "0.6 0.4" "0.4 0.6")


# Fixed parameters that work well
WEIGHT_DECAY="0.05"
EPOCHS=12
WARMUP_EPOCHS=2
MIN_LR="1e-8"

# Function to extract metrics from log file - IMPROVED VERSION
extract_metrics() {
    local log_file=$1
    
    # Extract validation accuracy - look for "Accuracy of the network on the val" pattern
    local val_acc=$(grep -E "Accuracy of the network on the .* val" $log_file | tail -1 | grep -oP '[0-9.]+(?=%)')
    if [ -z "$val_acc" ]; then
        # Try to find in a different format
        val_acc=$(grep -E "Max accuracy:" $log_file | grep -oP '[0-9.]+(?=%)')
    fi
    
    # Extract test accuracy 
    local test_acc=$(grep -E "Accuracy of the network on the .* test" $log_file | tail -1 | grep -oP '[0-9.]+(?=%)')
    if [ -z "$test_acc" ]; then
        # Try to extract from different format
        test_acc=$(grep -E "Acc@1" $log_file | tail -1 | awk '{print $2}')
    fi
    
    # Extract confusion matrix for sensitivity/specificity
    local sensitivity="0"
    local specificity="0"
    local f1_score="0"
    
    # First try direct sensitivity/specificity mentions
    if grep -q "Sensitivity:" $log_file; then
        sensitivity=$(grep "Sensitivity:" $log_file | tail -1 | grep -oP '[0-9.]+')
    fi
    
    if grep -q "specificity :" $log_file; then
        specificity=$(grep "specificity :" $log_file | tail -1 | grep -oP '[0-9.]+')
    fi
    
    # If not found, extract from confusion matrix
    if [[ "$sensitivity" == "0" || "$specificity" == "0" ]]; then
        # Find the confusion matrix section
        local cm_text=""
        if grep -q "\[\[" $log_file; then
            # Get line numbers where matrix appears
            local start_line=$(grep -n "\[\[" $log_file | head -1 | cut -d: -f1)
            if [ -n "$start_line" ]; then
                # Extract 2 lines of the matrix
                cm_text=$(sed -n "${start_line},$(($start_line+1))p" $log_file)
                
                # Extract values from matrix format [[TN FP] [FN TP]]
                local tn=$(echo "$cm_text" | head -1 | grep -oP '\[\[\s*\K[0-9]+')
                local fp=$(echo "$cm_text" | head -1 | grep -oP '\[\[\s*[0-9]+\s+\K[0-9]+')
                local fn=$(echo "$cm_text" | tail -1 | grep -oP '\[\s*\K[0-9]+')
                local tp=$(echo "$cm_text" | tail -1 | grep -oP '\[\s*[0-9]+\s+\K[0-9]+')
                
                if [[ -n "$tn" && -n "$fp" && -n "$fn" && -n "$tp" ]]; then
                    # Calculate metrics
                    sensitivity=$(echo "scale=4; $tp / ($tp + $fn)" | bc)
                    specificity=$(echo "scale=4; $tn / ($tn + $fp)" | bc)
                    
                    # Calculate precision and F1
                    local precision=$(echo "scale=4; $tp / ($tp + $fp)" | bc 2>/dev/null)
                    if [[ "$precision" != "0" && "$sensitivity" != "0" ]]; then
                        f1_score=$(echo "scale=4; 2 * $precision * $sensitivity / ($precision + $sensitivity)" | bc 2>/dev/null)
                    fi
                fi
            fi
        fi
    fi
    
    # Return metrics as CSV - ensure non-empty values
    val_acc=${val_acc:-"0"}
    test_acc=${test_acc:-"0"}
    sensitivity=${sensitivity:-"0"}
    specificity=${specificity:-"0"}
    f1_score=${f1_score:-"0"}
    
    echo "$val_acc,$test_acc,$sensitivity,$specificity,$f1_score"
}

# Track best performance
BEST_F1=0
BEST_BALANCED=0
BEST_RUN=0
BEST_LR=""
BEST_DP=""
BEST_CW=""
BEST_LD=""

# Run counter
run_id=0

# Grid search over learning rate, drop path, class weights, and layer decay
for lr in "${LEARNING_RATES[@]}"; do
    for dp in "${DROP_PATH_RATES[@]}"; do
        for cw in "${CLASS_WEIGHTS[@]}"; do
            for ld in "${LAYER_DECAYS[@]}"; do
                # Increment run counter
                run_id=$((run_id+1))
                
                echo "======= Run $run_id: LR=$lr, DP=$dp, CW=$cw, LD=$ld ======="
                
                # Clear GPU memory
                python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
                
                # Create run directory with descriptive name
                RUN_DIR="${RESULTS_DIR}/run_${run_id}_lr${lr}_dp${dp}_cw${cw//,/_}_ld${ld}"
                mkdir -p $RUN_DIR
                LOG_FILE="${RUN_DIR}/train.log"
                
                # Run training with optimized parameters
                torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=1 \
                ../run_class_finetuning.py \
                        --model vit_small_patch16_224 \
                        --data_set GBC_Net \
                        --nb_classes 2 \
                        --save_ckpt \
                        --data_path ${DATA_PATH} \
                        --finetune ${MODEL_PATH} \
                        --no_auto_resume \
                        --log_dir ${RUN_DIR} \
                        --output_dir ${RUN_DIR} \
                        --batch_size 1 \
                        --input_size 224 \
                        --short_side_size 224 \
                        --num_frames 16 \
                        --sampling_rate 2 \
                        --num_sample 2 \
                        --num_workers 2 \
                        --opt adamw \
                        --num_segment 1 \
                        --lr ${lr} \
                        --min_lr ${MIN_LR} \
                        --warmup_epochs ${WARMUP_EPOCHS} \
                        --opt_betas 0.9 0.999 \
                        --test_num_segment 10 \
                        --test_num_crop 3 \
                        --epochs ${EPOCHS} \
                        --test_randomization \
                        --dist_eval \
                        --enable_deepspeed \
                        --weight_decay ${WEIGHT_DECAY} \
                        --layer_decay ${ld} \
                        --update_freq 4 \
                        --drop_path ${dp} \
                        --class_weights ${cw} \
                        --aa rand-m7-mstd0.5-inc1 \
                        --mixup 0.6 \
                        --cutmix 0.7 \
                        --mixup_prob 0.7 \
                        --clip_grad 1.0 \
                        --save_ckpt_freq 100 \
                        2>&1 | tee ${LOG_FILE}
                
                # Extract metrics
                metrics=$(extract_metrics ${LOG_FILE})
                IFS=',' read -r val_acc test_acc sensitivity specificity f1_score <<< "$metrics"
                
                # Save results to CSV - carefully formatted to avoid the previous issues
                echo "$run_id,$lr,$dp,$cw,$ld,$val_acc,$test_acc,$sensitivity,$specificity,$f1_score" >> ${RESULTS_DIR}/results.csv
                
                # Calculate a balanced score (harmonic mean of sensitivity and specificity)
                balanced_score=0
                if [[ "$sensitivity" != "0" && "$specificity" != "0" ]]; then
                    balanced_score=$(echo "scale=4; 2 * $sensitivity * $specificity / ($sensitivity + $specificity)" | bc 2>/dev/null)
                fi
                
                # Check if this is the best run - prioritize balanced performance
                if (( $(echo "$balanced_score > $BEST_BALANCED" | bc -l) )); then
                    BEST_BALANCED=$balanced_score
                    BEST_F1=$f1_score
                    BEST_RUN=$run_id
                    BEST_LR=$lr
                    BEST_DP=$dp
                    BEST_CW=$cw
                    BEST_LD=$ld
                elif (( $(echo "$balanced_score == $BEST_BALANCED" | bc -l) )) && (( $(echo "$f1_score > $BEST_F1" | bc -l) )); then
                    BEST_F1=$f1_score
                    BEST_RUN=$run_id
                    BEST_LR=$lr
                    BEST_DP=$dp
                    BEST_CW=$cw
                    BEST_LD=$ld
                fi
                
                echo "Finished run $run_id: LR=$lr, DP=$dp, CW=$cw, LD=$ld"
                echo "Current best: Run $BEST_RUN (LR=$BEST_LR, DP=$BEST_DP, CW=$BEST_CW, LD=$BEST_LD)"
                echo "Best balanced score: $BEST_BALANCED, F1: $BEST_F1"
                echo "----------------------------------------"
                sleep 5
            done
        done
    done
done

# Create a detailed summary file
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
echo "========== IMPROVED PARAMETER GRID SEARCH SUMMARY ==========" > $SUMMARY_FILE
echo "Completed at: $(date)" >> $SUMMARY_FILE
echo -e "\nParameters tested:" >> $SUMMARY_FILE
echo "Learning Rates: ${LEARNING_RATES[*]}" >> $SUMMARY_FILE
echo "Drop Path Rates: ${DROP_PATH_RATES[*]}" >> $SUMMARY_FILE
echo "Class Weights: ${CLASS_WEIGHTS[*]}" >> $SUMMARY_FILE
echo "Layer Decay Values: ${LAYER_DECAYS[*]}" >> $SUMMARY_FILE
echo -e "\nFixed Parameters:" >> $SUMMARY_FILE
echo "Weight Decay: $WEIGHT_DECAY" >> $SUMMARY_FILE
echo "Epochs: $EPOCHS" >> $SUMMARY_FILE
echo "Warmup Epochs: $WARMUP_EPOCHS" >> $SUMMARY_FILE
echo "Min Learning Rate: $MIN_LR" >> $SUMMARY_FILE
echo -e "\nTotal configurations tested: $run_id" >> $SUMMARY_FILE

# Make sure we have a valid best run before processing
if [[ $BEST_RUN -gt 0 ]]; then
    # Find best run details from results.csv
    BEST_CONFIG=$(grep "^$BEST_RUN," ${RESULTS_DIR}/results.csv)
    
    # If found, parse the line
    if [[ -n "$BEST_CONFIG" ]]; then
        IFS=',' read -r run lr dp cw ld val_acc test_acc sensitivity specificity f1_score <<< "$BEST_CONFIG"
        
        echo -e "\nBest Configuration (Run $BEST_RUN):" >> $SUMMARY_FILE
        echo "Learning Rate: $lr" >> $SUMMARY_FILE
        echo "Drop Path Rate: $dp" >> $SUMMARY_FILE
        echo "Class Weights: $cw" >> $SUMMARY_FILE
        echo "Layer Decay: $ld" >> $SUMMARY_FILE
        echo -e "\nPerformance:" >> $SUMMARY_FILE
        echo "Validation Accuracy: $val_acc%" >> $SUMMARY_FILE
        echo "Test Accuracy: $test_acc%" >> $SUMMARY_FILE
        echo "Sensitivity: $sensitivity" >> $SUMMARY_FILE
        echo "Specificity: $specificity" >> $SUMMARY_FILE
        echo "F1 Score: $f1_score" >> $SUMMARY_FILE
        echo "Balanced Score (Harmonic mean of sens/spec): $BEST_BALANCED" >> $SUMMARY_FILE
    else
        echo -e "\nBest run ($BEST_RUN) not found in results.csv!" >> $SUMMARY_FILE
    fi
else
    echo -e "\nNo valid runs completed. Check logs for errors." >> $SUMMARY_FILE
fi

# Add comparative analysis by parameter
echo -e "\n======= Learning Rate Analysis =======" >> $SUMMARY_FILE
for lr in "${LEARNING_RATES[@]}"; do
    avg_bal=$(grep ",$lr," ${RESULTS_DIR}/results.csv | awk -F, '{sens=$8; spec=$9; if(sens>0 && spec>0) bal=2*sens*spec/(sens+spec); else bal=0; sum+=bal; n++} END {if(n>0) print sum/n; else print "N/A"}')
    echo "LR $lr: Average Balanced Score = $avg_bal" >> $SUMMARY_FILE
done

echo -e "\n======= Drop Path Analysis =======" >> $SUMMARY_FILE
for dp in "${DROP_PATH_RATES[@]}"; do
    avg_bal=$(grep ",$dp," ${RESULTS_DIR}/results.csv | awk -F, '{sens=$8; spec=$9; if(sens>0 && spec>0) bal=2*sens*spec/(sens+spec); else bal=0; sum+=bal; n++} END {if(n>0) print sum/n; else print "N/A"}')
    echo "DP $dp: Average Balanced Score = $avg_bal" >> $SUMMARY_FILE
done

echo -e "\n======= Class Weight Analysis =======" >> $SUMMARY_FILE
for cw in "${CLASS_WEIGHTS[@]}"; do
    avg_bal=$(grep ",$cw," ${RESULTS_DIR}/results.csv | awk -F, '{sens=$8; spec=$9; if(sens>0 && spec>0) bal=2*sens*spec/(sens+spec); else bal=0; sum+=bal; n++} END {if(n>0) print sum/n; else print "N/A"}')
    echo "CW $cw: Average Balanced Score = $avg_bal" >> $SUMMARY_FILE
done

# Create a script with the best configuration for future use
BEST_SCRIPT="${RESULTS_DIR}/train_best_config.sh"
cat > $BEST_SCRIPT << EOF
#!/bin/bash
# Best configuration from improved parameter grid search

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export MASTER_PORT=\${MASTER_PORT:-12320}
export OMP_NUM_THREADS=1

# Run training with best parameters
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=1 \\
../run_class_finetuning.py \\
        --model vit_small_patch16_224 \\
        --data_set GBC_Net \\
        --nb_classes 2 \\
        --save_ckpt \\
        --data_path ${DATA_PATH} \\
        --finetune ${MODEL_PATH} \\
        --no_auto_resume \\
        --log_dir ./output_best \\
        --output_dir ./output_best \\
        --batch_size 1 \\
        --input_size 224 \\
        --short_side_size 224 \\
        --num_frames 16 \\
        --sampling_rate 2 \\
        --num_sample 2 \\
        --num_workers 2 \\
        --opt adamw \\
        --num_segment 1 \\
        --lr ${BEST_LR} \\
        --min_lr ${MIN_LR} \\
        --warmup_epochs ${WARMUP_EPOCHS} \\
        --opt_betas 0.9 0.999 \\
        --test_num_segment 10 \\
        --test_num_crop 3 \\
        --epochs 30 \\
        --test_randomization \\
        --dist_eval \\
        --enable_deepspeed \\
        --weight_decay ${WEIGHT_DECAY} \\
        --layer_decay ${BEST_LD} \\
        --update_freq 4 \\
        --drop_path ${BEST_DP} \\
        --class_weights ${BEST_CW} \\
        --aa rand-m7-mstd0.5-inc1 \\
        --mixup 0.6 \\
        --cutmix 0.7 \\
        --mixup_prob 0.7 \\
        --clip_grad 1.0 \\
        --save_ckpt_freq 5
EOF
chmod +x $BEST_SCRIPT

echo "Grid search complete!"
echo "Results directory: ${RESULTS_DIR}"
echo "Summary file: ${SUMMARY_FILE}"
echo "Best configuration script: ${BEST_SCRIPT}"
