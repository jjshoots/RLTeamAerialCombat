#!/bin/bash

if [ -d .venv ]; then
  source .venv/bin/activate
else
  source venv/bin/activate
fi
wingman-compress-weights

rsync -rav --progress --stats "arctic-linx:~/Sandboxes/dogfighter/weights/$1" ./weights
python3 src/main.py --mode.display --model.id="$1"
