#!/bin/bash

if [ -d .venv ]; then
  source .venv/bin/activate
else
  source venv/bin/activate
fi
wingman-compress-weights

declare -a pids=()
wandb agent jjshoots/dogfighter/kbxzk5v8 &
pids+=($!)
sleep 10
wandb agent jjshoots/dogfighter/kbxzk5v8 &
pids+=($!)
sleep 10
wandb agent jjshoots/dogfighter/kbxzk5v8 &
pids+=($!)
sleep 10
wandb agent jjshoots/dogfighter/kbxzk5v8 &
pids+=($!)
sleep 10
if [ "$hostname" != "arctic-linx" ]; then
  # don't abuse my main PC
  wandb agent jjshoots/dogfighter/kbxzk5v8 &
  pids+=($!)
  sleep 10
  wandb agent jjshoots/dogfighter/kbxzk5v8 &
  pids+=($!)
  sleep 10
  wandb agent jjshoots/dogfighter/kbxzk5v8 &
  pids+=($!)
  sleep 10
  wandb agent jjshoots/dogfighter/kbxzk5v8 &
  pids+=($!)
  sleep 10
fi

for pid in ${pids[*]}; do
    wait $pid
done

# poweroff
