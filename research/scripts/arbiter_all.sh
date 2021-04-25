#!/usr/bin/env bash
# collect data for all envs in a tier.
# usage: bash scripts/arbiter_all.sh TIER
if [ -z "$1" ] || [ $1 -eq 0 ]; then
    envs=(Dropbox Bounce Bounce2 Object2)
elif [ $1 -eq 1 ]; then
    envs=(Urchin Luxo UrchinCube LuxoCube UrchinBall LuxoBall)
else
    echo "nothing"
    exit 1;
fi
for env in ${envs[@]}; do
    python -m research.main --mode=train --env=$env --datapath=logs/datadump/fill/comprehensive/$env/ --model=MultiStepArbiter --lr=5e-4 --log_n=1000 --bs=32 --nfilter=64 --hidden_size=256 --logdir=logs/april22/arbiter/$env/ --window=5 --total_itr=30000
done