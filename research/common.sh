# COMMON SHELL COMMANDS THAT I SAVE SO I DON'T FORGET THEM

# convert a set of images to a single video gif
convert -resize 100% -delay 2 -loop 0 *.png test.gif

# collect data 
python3 main.py --mode=collect --env=Urchin --collect_n=10000
# train
# multi-step
python main.py --mode=train --env=UrchinBall --datapath=$DP --model=multistep --vidstack=4 --phase=1 --log_n=50 --logdir=logs/biphase/x2/
python main.py --mode=train --env=UrchinBall --datapath=$DP --model=multistep --vidstack=4 --phase=2 --log_n=1 --logdir=logs/biphase/x2/phase2/1e5/nl3_512_16_32stacks/ --n_layer=3 --n_embed=512 --n_head=16 --bs=16 --amp=1

# frame_token
python main.py --mode=train --env=Luxo --datapath=$DP --model=frame_token --logdir=logs/luxo/flattie/ --lr=1e-3 --n_layer=3 --n_embed=512 --n_head=16 --lr=5e-4



# flat everything
# state vqvae so that state is discrete binary
python main.py --mode=train --env=Luxo --datapath=logs/datadump/luxo_2.0/ --wh_ratio=2.0 --model=statevq --window=16 --log_n=1000 --lr=1e-3 --logdir=logs/ternary/juststate128_512_save/ --bs=32 --log_n=1000 --lr=1e-3 --vqK=128 --hidden_size=512

# then flate everything model
python main.py --mode=train --env=Luxo --datapath=logs/datadump/luxo_2.0/ --wh_ratio=2.0 --model=flatev --log_n=1000 --lr=1e-3 --logdir=logs/flatev/med_bs32_ESR --bs=32 --log_n=1000 --lr=1e-3 --weightdir=logs/ternary/juststate128_512_save/ --window=200 --n_layer=3 --n_head=16 --hidden_size=512 --n_embed=512

# test learned simulator
python learned_env.py --env=Luxo --datapath=logs/datadump/big_luxo_2.0/ --wh_ratio=2.0 --model=flatev --log_n=1000 --lr=1e-3 --weightdir=logs/flatev/100window/MONSTER_bs64_6e-4 --bs=64 --log_n=1000 --lr=1e-3 --window=100 --n_layer=6 --n_head=32 --hidden_size=1024 --n_embed=1024 --lr=8e-4 --logdir=logs/trash/ --ipython_mode=1 --goals=1 --device=cuda --bs=4

 python learned_env.py --env=Luxo --datapath=logs/datadump/big_luxo_2.0/ --wh_ratio=2.0 --model=flatev --log_n=1000 --lr=1e-3 --weightdir=logs/flatev/100window/monsta2/  --goals=1 --num_envs=8 --window=100


# run RL on learned simulator env
 python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=flatev --use_done=0 --bs=100 --state_rew=1 --net=mlp --weightdir=logs/flatev/100window/smallnet_bs32_8e-4_fix/ --window=100 --goals=1 --num_envs=8 --lenv=1 --logdir=logs/rl/lenv/swap_8env_equalize_goals_m_temp_0.1_morenet_FIXMODEL8_lalp3 --lenv_temp=0.1 --bs=512 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --reset_prompt=0

 python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=flatev --state_rew=1 --net=mlp --weightdir=logs/flatev/100window/monsta2/ --window=100 --goals=1 --num_envs=8 --lenv=1 --logdir=logs/rl/lenv/bigger_net_temp0.1 --lenv_temp=0.1 --bs=512 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --reset_prompt=0


python main.py --mode=collect --env=UrchinCube --train_barrels=1000 --logdir=logs/datadump/urchin_cube
