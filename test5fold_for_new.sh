#!/usr/bin/env bash
set -x

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MASTER_PORT=${MASTER_PORT:-12320}
export OMP_NUM_THREADS=1


OUTPUT_DIR='../Dataset_folder/GBC_data/fold_3'
DATA_PATH='../Dataset_folder/GBC_data/folds/fold_3'
MODEL_PATH='../Dataset_folder/work_dir/pretrain_output/checkpoint_folder/checkpoint-149.pth'
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
        --auto_resume \
        --nb_classes 2 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --update_freq 2 \
        --input_size ${INPUT_SIZE} \
        --num_workers 10 \
        --sampling_rate 3 \
        --num_sample 4 \
        --num_workers 4 \
        --opt adamw \
        --lr 7e-4 \
        --min_lr 1e-8 \
        --num_segment 1 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --epochs 90 \
        --test_randomization \
        --dist_eval --enable_deepspeed \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}


OUTPUT_DIR='../Dataset_folder/GBC_data/fold_4'
DATA_PATH='../Dataset_folder/GBC_data/folds/fold_4'

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE}\
 ../run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set GBC_Net \
        --auto_resume \
        --nb_classes 2 \
        --save_ckpt \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --update_freq 4 \
        --input_size ${INPUT_SIZE} \
        --sampling_rate 3 \
        --num_sample 2 \
        --num_workers 4 \
        --opt adamw \
        --lr 7e-4 \
        --min_lr 1e-8 \
        --num_segment 1 \
        --test_num_segment 10 \
        --test_num_crop 3 \
        --epochs 59 \
        --test_randomization \
        --dist_eval --enable_deepspeed \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}



OUTPUT_DIR='../Dataset_folder/GBC_data/folds/fold_2'
DATA_PATH='../Dataset_folder/GBC_data/folds/fold_2'


torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE}\
 ../run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set GBC_Net \
        --auto_resume \
        --save_ckpt \
        --nb_classes 2 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --input_size ${INPUT_SIZE} \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 4 \
        --opt adamw \
        --lr 7e-4 \
        --min_lr 1e-8 \
        --num_segment 1 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --epochs 69 \
        --test_randomization \
        --dist_eval --enable_deepspeed \
        --update_freq 4 \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}



OUTPUT_DIR='../Dataset_folder/GBC_data/folds/fold_1'
DATA_PATH='../Dataset_folder/GBC_data/folds/fold_1'
#90 epochs

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE}\
 ../run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set GBC_Net \
        --auto_resume \
        --nb_classes 2 \
        --save_ckpt \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --input_size ${INPUT_SIZE} \
        --sampling_rate 3 \
        --num_sample 2 \
        --num_workers 4 \
        --opt adamw \
        --lr 7e-4 \
        --min_lr 1e-8 \
        --num_segment 1 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --epochs 59 \
        --test_randomization \
        --dist_eval --enable_deepspeed \
        --update_freq 4 \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}



OUTPUT_DIR='../Dataset_folder/GBC_data/fold_0'
DATA_PATH='../Dataset_folder/GBC_data/folds/fold_0'

#90 epcohs

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE}\
 ../run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set GBC_Net \
        --auto_resume \
        --nb_classes 2 \
        --save_ckpt \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --input_size ${INPUT_SIZE} \
        --sampling_rate 3 \
        --num_sample 2 \
        --num_workers 4 \
        --opt adamw \
        --lr 7e-4 \
        --min_lr 1e-8 \
        --num_segment 1 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --epochs 59 \
        --test_randomization \
        --dist_eval --enable_deepspeed \
        --update_freq 4 \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}
