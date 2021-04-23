#!/usr/bin/env bash
# collect data for all envs in a tier.
# usage: bash scripts/arbiter_all.sh TIER
if [ -z "$1" ] || [ $1 -eq 0 ]; then
    envs=(Dropbox Bounce Bounce2 Object2)
elif [ $1 -eq 1 ]; then
    envs=(Urchin Luxo UrchinCube LuxoCube UrchinBall LuxoBall)
else
    echo "nothing"
    exit 1
fi

for env in ${envs[@]}; do
    if [ $env == "Dropbox" ]; then
        window=25
        prompt_n=1
    else
        window=50
        prompt_n=3
    fi
    #python -m research.main --mode=train --env=$env --datapath=logs/datadump/fill/comprehensive/$env/ --arbiterdir=logs/april22/arbiter/$env --model=FlatBToken --lr=5e-4 --log_n=1000 --bs=32 --n_layer=2 --n_head=4 --n_embed=256 --hidden_size=256 --window=$window --total_itr=100000 --prompt_n=$prompt_n --logdir=logs/april22/video/FlatBToken/two/$env/ --weightdir=logs/april22a/autoencoder/BVAE/two/$env/
    python -m research.main --mode=train --env=$env --datapath=logs/datadump/fill/comprehensive/$env/ --arbiterdir=logs/april22/arbiter/$env --model=FlatBToken --lr=5e-4 --log_n=1000 --bs=32 --n_layer=4 --n_head=8 --n_embed=512 --hidden_size=512 --window=$window --total_itr=100000 --prompt_n=$prompt_n --logdir=logs/april22a/video/FlatBToken/$env/ --weightdir=logs/april22a/autoencoder/BVAE/$env/
done