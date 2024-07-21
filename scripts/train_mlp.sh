#!/bin/bash

source .venv/bin/activate
wingman-compress-weights

declare -a pids=()
python3 src/ma_env_main.py --train --wandb &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
