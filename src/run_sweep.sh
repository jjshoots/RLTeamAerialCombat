#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/obs_norm/ytg6s7md &
pids+=($!)
wandb agent jjshoots/obs_norm/ytg6s7md &
pids+=($!)
wandb agent jjshoots/obs_norm/ytg6s7md &
pids+=($!)
wandb agent jjshoots/obs_norm/ytg6s7md &
pids+=($!)

for pid in ${pids[*]}; do
  wait $pid
done
