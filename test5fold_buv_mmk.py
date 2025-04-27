#!/usr/bin/env bash
set -x

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MASTER_PORT=${MASTER_PORT:-12320}
export OMP_NUM_THREADS=1


OUTPUT_DIR='../Dataset_folder/work_dir/finetune_output/checkpoint_folder/spec-sens/buv_fold_5'
DATA_PATH='/home/hpc12/Lubaina/FocusMAE/Dataset_folder/BUV/cv_with_holdout/folds/fold_5'
MODEL_PATH='../Dataset_folder/work_dir/pretrain_output/checkpoint_folder/pre.pt'
COLUMN_NAME='pred_column'


N_NODES=${N_NODES:-1}
GPUS_PER_NODE=1
PY_ARGS=${@:3}
NODE_RANK=0
INPUT_SIZE=224


torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE} \
 ../run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set GBC_Net \
        --nb_classes 2 \
        --auto_resume \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR}\
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 2 \
        --opt adamw \
        --num_segment 1 \
        --lr 7e-4 \
        --class_weights 6 1 \
        --min_lr 1e-8 \
        --warmup_epochs 10 \
        --opt_betas 0.9 0.95 \
        --test_num_segment 20 \
        --test_num_crop 7 \
        --epochs 30 \
        --test_randomization \
        --dist_eval \
        --enable_deepspeed \
        --weight_decay 0.02 \
        --layer_decay 0.9 \
        --update_freq 8 \
        --drop_path 0.15 \
        --drop 0.1 \
        --aa rand-m7-mstd0.5-inc1 \
        --save_ckpt \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}



OUTPUT_DIR='../Dataset_folder/work_dir/finetune_output/checkpoint_folder/spec-sens/buv_fold_4'
DATA_PATH='/home/hpc12/Lubaina/FocusMAE/Dataset_folder/BUV/cv_with_holdout/folds/fold_4'

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE}\
 ../run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set GBC_Net \
        --nb_classes 2 \
        --auto_resume \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR}\
        --output_dir ${OUTPUT_DIR}\
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 2 \
        --opt adamw \
        --num_segment 1 \
        --lr 7e-4 \
        --class_weights 6 1 \
        --min_lr 1e-8 \
        --warmup_epochs 10 \
        --opt_betas 0.9 0.95 \
        --test_num_segment 20 \
        --test_num_crop 7 \
        --epochs 30 \
        --test_randomization \
        --dist_eval \
        --enable_deepspeed \
        --weight_decay 0.02 \
        --layer_decay 0.9 \
        --update_freq 8 \
        --drop_path 0.15 \
        --drop 0.1 \
        --aa rand-m7-mstd0.5-inc1 \
        --save_ckpt \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}



OUTPUT_DIR='../Dataset_folder/work_dir/finetune_output/checkpoint_folder/spec-sens/buv_fold_3'
DATA_PATH='/home/hpc12/Lubaina/FocusMAE/Dataset_folder/BUV/cv_with_holdout/folds/fold_3'

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE}\
 ../run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set GBC_Net \
        --nb_classes 2 \
        --auto_resume \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR}\
        --output_dir ${OUTPUT_DIR}\
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 2 \
        --opt adamw \
        --num_segment 1 \
        --lr 7e-4 \
        --class_weights 6 1 \
        --min_lr 1e-8 \
        --warmup_epochs 10 \
        --opt_betas 0.9 0.95 \
        --test_num_segment 20 \
        --test_num_crop 7 \
        --epochs 30 \
        --test_randomization \
        --dist_eval \
        --enable_deepspeed \
        --weight_decay 0.02 \
        --layer_decay 0.9 \
        --update_freq 8 \
        --drop_path 0.15 \
        --drop 0.1 \
        --aa rand-m7-mstd0.5-inc1 \
        --save_ckpt \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}



OUTPUT_DIR='../Dataset_folder/work_dir/finetune_output/checkpoint_folder/spec-sens/buv_fold_2'
DATA_PATH='/home/hpc12/Lubaina/FocusMAE/Dataset_folder/BUV/cv_with_holdout/folds/fold_2'

#90 epochs

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE}\
 ../run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set GBC_Net \
        --nb_classes 2 \
        --auto_resume \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR}\
        --output_dir ${OUTPUT_DIR}\
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 2 \
        --opt adamw \
        --num_segment 1 \
        --lr 7e-4 \
        --class_weights 6 1 \
        --min_lr 1e-8 \
        --warmup_epochs 10 \
        --opt_betas 0.9 0.95 \
        --test_num_segment 20 \
        --test_num_crop 7 \
        --epochs 30 \
        --test_randomization \
        --dist_eval \
        --enable_deepspeed \
        --weight_decay 0.02 \
        --layer_decay 0.9 \
        --update_freq 8 \
        --drop_path 0.15 \
        --drop 0.1 \
        --aa rand-m7-mstd0.5-inc1 \
        --save_ckpt \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}



OUTPUT_DIR='../Dataset_folder/work_dir/finetune_output/checkpoint_folder/spec-sens/buv_fold_1'
DATA_PATH='/home/hpc12/Lubaina/FocusMAE/Dataset_folder/BUV/cv_with_holdout/folds/fold_1'


#90 epcohs

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE}\
 ../run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set GBC_Net \
        --nb_classes 2 \
        --auto_resume \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR}\
        --output_dir ${OUTPUT_DIR}\
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 2 \
        --opt adamw \
        --num_segment 1 \
        --lr 7e-4 \
        --class_weights 6 1 \
        --min_lr 1e-8 \
        --warmup_epochs 10 \
        --opt_betas 0.9 0.95 \
        --test_num_segment 20 \
        --test_num_crop 7 \
        --epochs 30 \
        --test_randomization \
        --dist_eval \
        --enable_deepspeed \
        --weight_decay 0.02 \
        --layer_decay 0.9 \
        --update_freq 8 \
        --drop_path 0.15 \
        --drop 0.1 \
        --aa rand-m7-mstd0.5-inc1 \
        --save_ckpt \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}
