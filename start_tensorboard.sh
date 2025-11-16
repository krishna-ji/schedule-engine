#!/bin/bash
# Start TensorBoard for RL Training
# Usage: ./start_tensorboard.sh

echo -e "\033[32mStarting TensorBoard...\033[0m"
echo -e "\033[36mLog directory: logs/tensorboard/train\033[0m"
echo ""
echo -e "\033[33mTensorBoard will be available at:\033[0m"
echo -e "\033[32m  http://localhost:6006/\033[0m"
echo ""
echo -e "\033[90mPress Ctrl+C to stop TensorBoard\033[0m"
echo ""

uv run tensorboard --logdir logs/tensorboard/train --port 6006 --bind_all
