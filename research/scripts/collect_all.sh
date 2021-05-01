#!/usr/bin/env bash
# collect data for all envs in a tier.
# usage: bash scripts/collect_all.sh 

args="python -m research.main --mode=collect --num_envs=10 --train_barrels=100 --test_barrels=10"
datapath="logs/trash"
$args --env=Dropbox --logdir=$datapath/Dropbox
$args --env=Bounce --logdir=$datapath/Bounce
$args --env=Bounce2 --logdir=$datapath/Bounce2
$args --env=Object2 --logdir=$datapath/Object2
$args --env=Urchin --logdir=$datapath/Urchin
$args --env=Luxo --logdir=$datapath/Luxo
$args --env=UrchinCube --logdir=$datapath/UrchinCube
$args --env=LuxoCube --logdir=$datapath/LuxoCube
$args --env=UrchinBall --logdir=$datapath/UrchinBall
$args --env=LuxoBall --logdir=$datapath/LuxoBall
