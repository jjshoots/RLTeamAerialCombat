#!/bin/bash

source venv/bin/activate

declare -a pids=()

wandb agent jjshoots/obs_norm/7k8zkrdh &
pids+=($!)
wandb agent jjshoots/obs_norm/7k8zkrdh &
pids+=($!)
wandb agent jjshoots/obs_norm/7k8zkrdh &
pids+=($!)
wandb agent jjshoots/obs_norm/7k8zkrdh &
pids+=($!)

for pid in ${pids[*]}; do
  wait $pid
done
