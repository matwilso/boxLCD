#python3 rl/ppo.py --env=Luxo --goals=1 --num_envs=12 --bs=4096 --hidden_size=256 --logdir=logs/april24/rl/ppo/april27a/Luxo_real/ --total_steps=500000 --goal_thresh=0.05

python3 rl/ppo.py --env=Urchin --goals=1 --num_envs=12 --bs=4096 --hidden_size=256 --logdir=logs/april24/rl/ppo/april27a/Urchin_real_1M/ --total_steps=1000000 --goal_thresh=0.05

#python3 rl/ppo.py --env=Luxo --model=FlatRonald --weightdir=logs/april22a/video/FlatRonald/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april24/rl/ppo/april27a/Luxo_lenv --lenv_temp=1.0 --bs=4096 --hidden_size=256 --total_steps=500000 --goal_thres=0.05

python3 rl/ppo.py --env=Urchin --model=FlatRonald --weightdir=logs/april22a/video/FlatRonald/Urchin/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april24/rl/ppo/april27a/Urchin_lenv_1M --lenv_temp=1.0 --bs=4096 --hidden_size=256 --goal_thres=0.05 --total_steps=1000000
