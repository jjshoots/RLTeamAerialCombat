#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/obs_norm/emyu2rfs &
pids+=($!)
wandb agent jjshoots/obs_norm/emyu2rfs &
pids+=($!)
wandb agent jjshoots/obs_norm/emyu2rfs &
pids+=($!)
wandb agent jjshoots/obs_norm/emyu2rfs &
pids+=($!)

for pid in ${pids[*]}; do
  wait $pid
done
