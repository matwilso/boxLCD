#!/usr/bin/env bash
# collect data for all envs in a tier.
# usage: bash scripts/arbiter_all.sh TIER
if [ -z "$1" ] || [ $1 -eq 0 ]; then
    envs=(Dropbox Bounce Bounce2 Object3)
elif [ $1 -eq 1 ]; then
    envs=(Urchin Luxo UrchinCube LuxoCube UrchinBall LuxoBall)
else
    echo "nothing"
    exit 1
fi

for env in ${envs[@]}; do
    if [ $env == "Dropbox" ]; then
        window = 25
        prompt_n = 1
    else
        window = 50
        prompt_n = 3
    fi
    python -m research.main --mode=train --env=$env --datapath=logs/datadump/$env/ --arbiterdir=logs/april30/arbiter/$env --model=RSSM --lr=5e-4 --log_n=1000 --bs=32 --nfilter=64 --hidden_size=300 --window=$window --total_itr=100000 --free_nats=0.01 --prompt_n=$prompt_n --logdir=logs/april30/video/RSSM/$env/
done
