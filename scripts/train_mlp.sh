#!/bin/bash

source .venv/bin/activate
wingman-compress-weights

declare -a pids=()
python3 src/train_mlp_env.py --train --wandb &
pids+=($!)
sleep 10
python3 src/train_mlp_env.py --train --wandb &
pids+=($!)
sleep 20

for pid in ${pids[*]}; do
    wait $pid
done
