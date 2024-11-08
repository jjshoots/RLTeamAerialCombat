#!/bin/bash

if [ -d .venv ]; then
  source .venv/bin/activate
else
  source venv/bin/activate
fi
wingman-compress-weights

declare -a pids=()
wandb agent jjshoots/dogfighter/w5cxiy6v &
pids+=($!)
sleep 10
wandb agent jjshoots/dogfighter/w5cxiy6v &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
