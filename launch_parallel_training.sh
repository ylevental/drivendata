#!/bin/bash
# Parallel training launcher - trains 3 models on 3 GPUs simultaneously

echo "============================================================"
echo "LAUNCHING PARALLEL TRAINING"
echo "============================================================"

# Kill any existing training processes
pkill -f "train_single_model.py"

# Train each model on separate GPU in background
echo "Starting EfficientNet-B3 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python3 train_single_model.py efficientnet_b3 > log_b3.txt 2>&1 &
PID_B3=$!

echo "Starting EfficientNet-B4 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python3 train_single_model.py efficientnet_b4 > log_b4.txt 2>&1 &
PID_B4=$!

echo "Starting ResNet50 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python3 train_single_model.py resnet50 > log_resnet.txt 2>&1 &
PID_RESNET=$!

echo ""
echo "✓ All 3 models started in parallel!"
echo ""
echo "Process IDs:"
echo "  EfficientNet-B3 (GPU 0): $PID_B3"
echo "  EfficientNet-B4 (GPU 1): $PID_B4"
echo "  ResNet50 (GPU 2): $PID_RESNET"
echo ""
echo "Monitor progress:"
echo "  watch -n 1 nvidia-smi           # Watch GPU usage"
echo "  tail -f log_b3.txt              # Watch B3 training"
echo "  tail -f log_b4.txt              # Watch B4 training"
echo "  tail -f log_resnet.txt          # Watch ResNet training"
echo ""
echo "Wait for all to complete, then run predictions:"
echo "  python3 generate_ensemble_predictions.py"
echo ""
echo "============================================================"
