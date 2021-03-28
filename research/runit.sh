if [ $1 == 0 ]; then
    args="--env=Luxo --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=100 --use_done=0 --wh_ratio=2.0 --state_rew=1 --net=mlp --epochs=50"
    python rl/sac.py $args --logdir=logs/rl/exp/3e-4 --lr=3e-4
    python rl/sac.py $args --logdir=logs/rl/exp/3e-4_done --lr=3e-4 --use_done=1
    python rl/sac.py $args --logdir=logs/rl/exp/3e-4_lalp1e-4 --lr=3e-4 --alpha_lr=1e-4
    python rl/sac.py $args --logdir=logs/rl/exp/1e-3_lalp1e-4 --lr=1e-3 --alpha_lr=1e-4
    python rl/sac.py $args --logdir=logs/rl/exp/5e-4_lalp1e-4 --lr=5e-4 --alpha_lr=1e-4
elif [ $1 == 1 ]; then
    #python rl/sac.py --env=Luxo --logdir=logs/rl/exp/state_based/bs1024/cnn_delt --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=1024 --use_done=0 --wh_ratio=2.0 --net=cnn --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/state_based/bs512/cmlp_delt --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cmlp --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/state_based/bs512/cmlp_delt_5e-4 --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cmlp --epochs=50 --lr=5e-4
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/state_based/bs512/cnn_delt_5e-4 --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cnn --epochs=50 --lr=5e-4

    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/state_based/bs512/cnn_delt --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cnn --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/state_based/bs512/cmlp_delt --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cmlp --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/state_based/bs512/cmlp_delt_5e-4 --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cmlp --epochs=50 --lr=5e-4
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/state_based/bs512/cnn_delt_5e-4 --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cnn --epochs=50 --lr=5e-4

    #python rl/sac.py --env=Luxo --logdir=logs/rl/exp/state_based/fix/cmlp_cat --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cmlp --epochs=50 --zdelta=0
    #python rl/sac.py --env=Luxo --logdir=logs/rl/exp/state_based/fix/cnn_cat --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cnn --epochs=50 --zdelta=0
    #python rl/sac.py --env=Luxo --logdir=logs/rl/exp/state_based/fix/cmlp_cat_5e-4 --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cmlp --epochs=50 --lr=5e-4 --zdelta=0
    #python rl/sac.py --env=Luxo --logdir=logs/rl/exp/state_based/fix/cnn_cat_5e-4 --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cnn --epochs=50 --lr=5e-4 --zdelta=0

elif [ $1 == 2 ]; then
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_1e-3 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cnn --alpha_lr=3e-4 --lr=1e-3 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_5e-4 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cnn --alpha_lr=3e-4 --lr=5e-4 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_1e-4 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cnn --alpha_lr=3e-4 --lr=1e-4 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_5e-5 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cnn --alpha_lr=3e-4 --lr=5e-5 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_1e-4_1024 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=1024 --use_done=0 --wh_ratio=2.0 --net=cnn --alpha_lr=3e-4 --lr=1e-4 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_1e-5 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0 --net=cnn --alpha_lr=3e-4 --lr=1e-5 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_1e-4_2048 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=2048 --use_done=0 --wh_ratio=2.0 --net=cnn --alpha_lr=3e-4 --lr=1e-4 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_3e-4_2048 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=2048 --use_done=0 --wh_ratio=2.0 --net=cnn --alpha_lr=3e-4 --lr=3e-4 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_5e-4_4096 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=4096 --use_done=0 --wh_ratio=2.0 --net=cnn --alpha_lr=3e-4 --lr=1e-4 --epochs=50

elif [ $1 == 3 ]; then
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/img/mlp --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=100 --use_done=0 --wh_ratio=2.0 --net=mlp --epochs=50 --state_rew=-0
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/img/cmlp --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=100 --use_done=0 --wh_ratio=2.0 --net=cmlp --epochs=50 --state_rew=0
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/img/cnn --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=100 --use_done=0 --wh_ratio=2.0 --net=cnn --epochs=50 --state_rew=0
elif [ $1 == 4 ]; then
    args="python3 rl/sac.py --env=Luxo --ep_len=200 --hidden_size=256 --use_done=0 --wh_ratio=2.0 --epochs=50 --state_rew=0 --net=cnn"
    #$args --logdir=logs/rl/image_based/cnn_lalp5e-5_lr3e-4_256 --learned_alpha=1 --lr=3e-4 --alpha_lr=5e-5 --bs=256
    #$args --logdir=logs/rl/image_based/cnn_lalp5e-5_lr3e-4_512 --learned_alpha=1 --lr=3e-4 --alpha_lr=5e-5 --bs=512
    #$args --logdir=logs/rl/image_based/cnn_nolap0.2_lr3e-4_512 --learned_alpha=0 --lr=3e-4 --alpha=0.2 --bs=512
    #$args --logdir=logs/rl/image_based/cnn_nolap0.2_lr1e-4_512 --learned_alpha=0 --lr=1e-4 --alpha=0.2 --bs=512
    #$args --logdir=logs/rl/image_based/cnn_nolap0.2_lr1e-3_1024 --learned_alpha=0 --lr=1e-3 --alpha=0.2 --bs=1024
    #$args --logdir=logs/rl/image_based/cnn_lalp1e-4_lr3e-4_512 --learned_alpha=1 --lr=3e-4 --alpha_lr=1e-4 --bs=512
    #$args --logdir=logs/rl/image_based/cnn_nolap0.2_lr3e-4_512_12env --learned_alpha=0 --lr=3e-4 --alpha=0.2 --bs=512 --num_envs=12
    $args --logdir=logs/rl/image_based/cnn_nolap0.2_lr3e-4_1024_12env --learned_alpha=0 --lr=3e-4 --alpha=0.2 --bs=1024 --num_envs=12
    #python rl/sac.py --env=Luxo --ep_len=200 --hidden_size=256 --use_done=0 --wh_ratio=2.0 --state_rew=0 --net=cnn --logdir=logs/rl/image_based/mlp_nolap0.2_lr3e-4_512_nenv12_1e-3 --num_envs=12 --learned_alpha=0 --lr=1e-3 --alpha=0.2 --bs=512
elif [ $1 == 4 ]; then
    python main.py --mode=train --env=Luxo --datapath=logs/datadump/luxo_2.0/ --wh_ratio=2.0 --model=vae --window=16 --logdir=logs/vaes/x2_beta0.25_5e-4/ --log_n=1000 --beta=0.25 --lr=5e-4 --total_itr=5000
elif [ $1 == 5 ]; then
    python main.py --mode=train --env=Luxo --datapath=logs/datadump/luxo_2.0/ --wh_ratio=2.0 --model=vae --window=16 --logdir=logs/vaes/x2_beta0.1_1e-3_bigger128_bs32/ --log_n=1000 --beta=0.1 --lr=1e-3 --refresh_data=1 --nfilter=128 --bs=32
else
    echo "null"
fi
