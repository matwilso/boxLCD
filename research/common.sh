# COMMON SHELL COMMANDS THAT I SAVE SO I DON'T FORGET THEM

# convert a set of images to a single video gif
convert -resize 100% -delay 2 -loop 0 *.png test.gif

# collect data
python main.py --mode=collect --env=UrchinCube --train_barrels=1000 --logdir=logs/datadump/urchin_cube

# MODEL LEARNING
# multi-step model
python main.py --mode=train --env=UrchinBall --datapath=$DP --model=multistep --vidstack=4 --phase=1 --log_n=50 --logdir=logs/biphase/x2/
python main.py --mode=train --env=UrchinBall --datapath=$DP --model=multistep --vidstack=4 --phase=2 --log_n=1 --logdir=logs/biphase/x2/phase2/1e5/nl3_512_16_32stacks/ --n_layer=3 --n_embed=512 --n_head=16 --bs=16 --amp=1

# frame_token
python main.py --mode=train --env=Luxo --datapath=$DP --model=frame_token --logdir=logs/luxo/flattie/ --lr=1e-3 --n_layer=3 --n_embed=512 --n_head=16 --lr=5e-4

# FLAT EVERYTHING (GOOD ONE)
# state vqvae so that state is discrete binary
python main.py --mode=train --env=Luxo --datapath=logs/datadump/luxo_2.0/ --wh_ratio=2.0 --model=statevq --window=16 --log_n=1000 --lr=1e-3 --logdir=logs/ternary/juststate128_512_save/ --bs=32 --log_n=1000 --lr=1e-3 --vqK=128 --hidden_size=512
# then flat everything model
python main.py --mode=train --env=Luxo --datapath=logs/datadump/luxo_2.0/ --wh_ratio=2.0 --model=flatev --log_n=1000 --lr=1e-3 --logdir=logs/flatev/med_bs32_ESR --bs=32 --log_n=1000 --lr=1e-3 --weightdir=logs/ternary/juststate128_512_save/ --window=200 --n_layer=3 --n_head=16 --hidden_size=512 --n_embed=512

# RL
python rl/sac.py --env=Luxo --wh_ratio=2.0 --bs=512 --hidden_size=512 --net=mlp --logdir=logs/rl/luxo
python rl/sac.py --env=Urchin --wh_ratio=2.0 --bs=512 --hidden_size=512 --net=mlp --logdir=logs/rl/urchin
python rl/sac.py --env=Luxo --wh_ratio=2.0 --bs=512 --hidden_size=512 --net=cnn --logdir=logs/rl/luxo_cnn
python rl/sac.py --env=Urchin --wh_ratio=2.0 --bs=512 --hidden_size=512 --net=cnn --logdir=logs/rl/urchin_cnn

# LEARNED SIMULATION
# test learned simulator
python learned_env.py --env=Luxo --datapath=logs/datadump/big_luxo_2.0/ --wh_ratio=2.0 --model=flatev --log_n=1000 --lr=1e-3 --weightdir=logs/flatev/x/monsta2/ --goals=1 --num_envs=8 --window=100
# run RL on learned simulator env, and non-learned env
python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=flatev --weightdir=logs/flatev/x/ --window=100 --goals=1 --num_envs=8 --lenv=1 --logdir=logs/rl/lenv/x --lenv_temp=0.1 --bs=512 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --reset_prompt=0

# CUBES
python rl/sac.py --env=UrchinCube --state_rew=1 --net=mlp --goals=1 --num_envs=8 --lenv=0 --bs=512 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=2.0
python rl/sac.py --env=CrabCube --state_rew=1 --net=mlp --goals=1 --num_envs=8 --lenv=0 --bs=512 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=2.0

# these 2 worked
python rl/sac.py --env=UrchinCube --state_rew=1 --net=mlp --goals=1 --num_envs=8 --lenv=0 --logdir=logs/rl/urchin_cube/10fps/diffdelt_1.5_2 --bs=128 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=1.5 --diff_delt=1 --fps=10
python rl/sac.py --env=UrchinCube --state_rew=1 --net=mlp --goals=1 --num_envs=8 --lenv=0 --logdir=logs/rl/urchin_cube/10fps/nodiffdelt_1.5 --bs=128 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=1.5 --diff_delt=0 --fps=10

# same
python rl/sac.py --env=UrchinCube --state_rew=1 --net=mlp --goals=1 --num_envs=8 --lenv=0 --logdir=logs/rl/urchin_cube/10fps/diffdelt_2.0_objchanges_halfmass --bs=128 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=2.0 --diff_delt=1 --fps=10
#python rl/sac.py --env=UrchinCube --state_rew=1 --net=mlp --goals=1 --num_envs=8 --lenv=0 --logdir=logs/rl/urchin_cube/10fps/diffdelt_1.5_objchanges_halfmass --bs=128 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=1.5 --diff_delt=1 --fps=10
