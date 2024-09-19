#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/obs_norm/u3gli8q4 &
pids+=($!)
wandb agent jjshoots/obs_norm/u3gli8q4 &
pids+=($!)
wandb agent jjshoots/obs_norm/u3gli8q4 &
pids+=($!)
wandb agent jjshoots/obs_norm/u3gli8q4 &
pids+=($!)

for pid in ${pids[*]}; do
  wait $pid
done
