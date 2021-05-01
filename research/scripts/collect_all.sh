#!/usr/bin/env bash
# collect data for all envs in a tier.
# usage: bash scripts/collect_all.sh TIER
# TODO: make these be in python since it is easier to quickly write python.
 
if [ -z "$1" ] || [ $1 -eq 0 ]; then
    envs=(Dropbox Bounce Bounce2 Object2)
elif [ $1 -eq 1 ]; then
    envs=(Urchin Luxo UrchinCube LuxoCube UrchinBall LuxoBall)
else
    echo "nothing"
    exit 1;
fi
for env in ${envs[@]}; do
    python -m research.main --mode=collect --num_envs=10 --train_barrels=100 --test_barrels=10 --env=$env --logdir=logs/datadump/$env
done
