#python rl/ppo.py --env=Luxo --goals=1 --num_envs=12 --bs=512 --hidden_size=256 --logdir=logs/april24/rl/ppo/bs512/ --pi_lr=3e-4 --vf_lr=1e-3 --total_steps=100000 --goal_thresh=0.005
#python rl/ppo.py --env=Luxo --goals=1 --num_envs=12 --bs=1024 --hidden_size=256 --logdir=logs/april24/rl/ppo/bs1024/ --pi_lr=3e-4 --vf_lr=1e-3 --total_steps=100000 --goal_thresh=0.005
#python rl/sac.py --env=Luxo --bs=128 --hidden_size=256 --net=mlp --num_envs=12 --logdir=logs/april24/rl/ppo/sac/x --goals=1 --goal_thresh=0.005

#python rl/sac.py --env=Luxo --model=FRNLD --weightdir=logs/april22a/video/FRNLD/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april23/rl/luxo/lenv/fix/ronald/0.005/0.5temp --lenv_temp=0.5 --bs=256 --hidden_size=512 --reset_prompt=1 --succ_reset=0 --total_steps=100000 --goal_thres=0.005
#
#python rl/sac.py --env=Luxo --model=FBT --weightdir=logs/april22a/video/FBT/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april23/rl/luxo/lenv/fix/btoken/0.005/0.1temp --lenv_temp=0.1 --bs=256 --hidden_size=512 --reset_prompt=1 --succ_reset=0 --total_steps=100000 --goal_thres=0.005
#
#python rl/sac.py --env=Luxo --model=FBT --weightdir=logs/april22a/video/FBT/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april23/rl/luxo/lenv/fix/btoken/0.005/0.5temp --lenv_temp=0.5 --bs=256 --hidden_size=512 --reset_prompt=1 --succ_reset=0 --total_steps=100000 --goal_thres=0.005
#
python rl/sac.py --env=Luxo --model=FRNLD --weightdir=logs/april22a/video/FRNLD/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april23/rl/luxo/lenv/fix/ronald/0.005/1.2temp_again --lenv_temp=1.2 --bs=256 --hidden_size=512 --reset_prompt=1 --succ_reset=0 --total_steps=100000 --goal_thres=0.005
python rl/sac.py --env=Luxo --model=FRNLD --weightdir=logs/april22a/video/FRNLD/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april23/rl/luxo/lenv/fix/ronald/0.005/0.1temp_again --lenv_temp=0.1 --bs=256 --hidden_size=512 --reset_prompt=1 --succ_reset=0 --total_steps=100000 --goal_thres=0.005
#
#
##python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=FRNLD --weightdir=logs/april22a/video/FRNLD/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april23/rl/luxo/lenv/ronald/1.0 --lenv_temp=1.0 --bs=256 --hidden_size=512 --reset_prompt=0 --succ_reset=0 --total_steps=100000
#
#python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=FBT --weightdir=logs/april22a/video/FBT/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april23/rl/luxo/lenv/btoken/0.005/0.1_bigger_bs256rp --lenv_temp=0.1 --bs=256 --hidden_size=512 --reset_prompt=1 --succ_reset=0 --total_steps=80000 --goal_thres=0.005
#
#python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=FRNLD --weightdir=logs/april22a/video/FRNLD/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april23/rl/luxo/lenv/ronald/0.005/1.0rp --lenv_temp=1.0 --bs=256 --hidden_size=512 --reset_prompt=1 --succ_reset=0 --total_steps=800000 --goal_thres=0.005
#
#
#python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=FBT --weightdir=logs/april22a/video/FBT/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april23/rl/luxo/lenv/btoken/0.005/0.1_bigger_bs512rp --lenv_temp=0.1 --bs=512 --hidden_size=512 --reset_prompt=1 --succ_reset=0 --total_steps=50000 --goal_thres=0.005
#
#python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=FBT --weightdir=logs/april22a/video/FBT/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april23/rl/luxo/lenv/btoken/0.005/0.5_bigger_bs256rp --lenv_temp=0.5 --bs=256 --hidden_size=512 --reset_prompt=1 --succ_reset=0 --total_steps=50000 --goal_thres=0.005
#
#python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=FBT --weightdir=logs/april22a/video/FBT/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april23/rl/luxo/lenv/btoken/0.005/1.0_bigger_bs256rp --lenv_temp=1.0 --bs=256 --hidden_size=512 --reset_prompt=1 --succ_reset=0 --total_steps=50000 --goal_thres=0.005
#
#python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=FRNLD --weightdir=logs/april22a/video/FRNLD/Luxo/ --window=50 --goals=1 --num_envs=12 --lenv=1 --logdir=logs/april23/rl/luxo/lenv/ronald/0.005/0.5rp --lenv_temp=0.5 --bs=256 --hidden_size=512 --reset_prompt=1 --succ_reset=0 --total_steps=50000 --goal_thres=0.005

