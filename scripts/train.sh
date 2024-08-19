#!/bin/bash

if [ -d .venv ]; then
  source .venv/bin/activate
else
  source venv/bin/activate
fi
wingman-compress-weights

declare -a pids=()
python3 src/main.py --mode.train --wandb.enable &
pids+=($!)
sleep 10
declare -a pids=()
python3 src/main.py --mode.train --wandb.enable &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
