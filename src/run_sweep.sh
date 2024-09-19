#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/obs_norm/bf00af52 &
pids+=($!)
wandb agent jjshoots/obs_norm/bf00af52 &
pids+=($!)
wandb agent jjshoots/obs_norm/bf00af52 &
pids+=($!)
wandb agent jjshoots/obs_norm/bf00af52 &
pids+=($!)

for pid in ${pids[*]}; do
  wait $pid
done
