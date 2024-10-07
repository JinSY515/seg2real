# accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=3 train.py
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
accelerate launch --multi_gpu --num_processes 2 --main_process_port=12223 train_w_controlNeXt.py