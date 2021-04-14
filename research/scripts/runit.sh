if [ $1 == 0 ]; then
    # test cnn with more entropy bonus throughout.
    python rl/sac.py --env=UrchinCube --state_rew=1 --net=cnn --goals=1 --num_envs=8 --lenv=0 --logdir=logs/rl/urchin_cube/10fps/diffdelt_2.0_objchanges_halfmass_cnn_alp0.2 --bs=128 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --state_key=full_state --use_done=0 --wh_ratio=2.0 --diff_delt=1 --fps=10 --alpha=0.2 --learned_alpha=0 --epochs=100
elif [ $1 == 1 ]; then
    python rl/sac.py --env=Luxo --wh_ratio=2.0 --model=flatev --weightdir=logs/flatev/10fps/luxo_100_8e-4/ --window=100 --goals=1 --num_envs=8 --lenv=0 --logdir=logs/rl/lenv/10fps/two/cnn64/luxo_real --lenv_temp=0.1 --bs=64 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --reset_prompt=0 --succ_reset=0 --goal_thresh=0.005 --net=cnn

    python rl/sac.py --env=Urchin --wh_ratio=2.0 --model=flatev --weightdir=logs/flatev/10fps/urchin_100/ --window=100 --goals=1 --num_envs=8 --lenv=0 --logdir=logs/rl/lenv/10fps/two/cnn64/urchin_real --lenv_temp=0.1 --bs=64 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --reset_prompt=0 --succ_reset=0 --goal_thresh=0.01 --net=cnn

    python rl/sac.py --env=Urchin --wh_ratio=2.0 --model=flatev --weightdir=logs/flatev/10fps/urchin_100/ --window=100 --goals=1 --num_envs=8 --lenv=0 --logdir=logs/rl/lenv/10fps/two/cnn64/urchin_real_alp0.2 --lenv_temp=0.1 --bs=64 --hidden_size=512 --learned_alpha=1 --alpha_lr=1e-4 --reset_prompt=0 --succ_reset=0 --goal_thresh=0.01 --net=cnn --learned_alpha=0 --alpha=0.2
else
    echo "null"
fi
