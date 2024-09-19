#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/obs_norm/dyrp2wxr &
pids+=($!)
wandb agent jjshoots/obs_norm/dyrp2wxr &
pids+=($!)
wandb agent jjshoots/obs_norm/dyrp2wxr &
pids+=($!)
wandb agent jjshoots/obs_norm/dyrp2wxr &
pids+=($!)

for pid in ${pids[*]}; do
  wait $pid
done
