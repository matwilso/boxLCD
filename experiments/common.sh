# COLLECT DATA 
python3 main.py --mode=collect --env=dropbox --collect_n=10000
# TRAIN
python3 main.py --mode=world --env=dropbox --datapath=$DP --logdir=logs/train/
