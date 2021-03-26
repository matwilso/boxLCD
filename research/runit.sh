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
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_1e-3 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0  --net=cnn --alpha_lr=3e-4 --lr=1e-3 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_5e-4 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0  --net=cnn --alpha_lr=3e-4 --lr=5e-4 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_1e-4 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0  --net=cnn --alpha_lr=3e-4 --lr=1e-4 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_5e-5 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0  --net=cnn --alpha_lr=3e-4 --lr=5e-5 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_1e-4_1024 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=1024 --use_done=0 --wh_ratio=2.0  --net=cnn --alpha_lr=3e-4 --lr=1e-4 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_1e-5 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=512 --use_done=0 --wh_ratio=2.0  --net=cnn --alpha_lr=3e-4 --lr=1e-5 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_1e-4_2048 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=2048 --use_done=0 --wh_ratio=2.0  --net=cnn --alpha_lr=3e-4 --lr=1e-4 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_3e-4_2048 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=2048 --use_done=0 --wh_ratio=2.0  --net=cnn --alpha_lr=3e-4 --lr=3e-4 --epochs=50
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/fixed_goal/linrew_cnn_5e-4_4096 --learned_alpha=1 --ep_len=200 --hidden_size=256 --bs=4096 --use_done=0 --wh_ratio=2.0  --net=cnn --alpha_lr=3e-4 --lr=1e-4 --epochs=50

elif [ $1 == 3 ]; then
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/img/mlp --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=100 --use_done=0 --wh_ratio=2.0 --net=mlp --epochs=50 --state_rew=-0
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/img/cmlp --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=100 --use_done=0 --wh_ratio=2.0 --net=cmlp --epochs=50 --state_rew=0
    python rl/sac.py --env=Luxo --logdir=logs/rl/exp/img/cnn --learned_alpha=1 --ep_len=500 --hidden_size=256 --bs=100 --use_done=0 --wh_ratio=2.0 --net=cnn --epochs=50 --state_rew=0
else
    echo "null"
fi