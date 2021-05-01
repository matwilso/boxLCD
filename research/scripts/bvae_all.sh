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
    python -m research.main --mode=train --env=$env --datapath=logs/datadump/$env/ --model=BVAE --lr=5e-4 --log_n=1000 --bs=32 --hidden_size=64 --vqK=64 --nfilter=16 --vqD=16 --window=5 --total_itr=30000 --logdir=logs/autoencoder/BVAE/$env/ 
done