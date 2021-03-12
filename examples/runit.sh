#python collect.py --env=Dropbox --collect_n=100000 &
#python collect.py --env=Bounce --collect_n=100000 &
#python collect.py --env=Urchin --collect_n=100000 & 
#python collect.py --env=UrchinBall --collect_n=100000 & 
#python collect.py --env=UrchinBalls --collect_n=100000 &
#python collect.py --env=UrchinCubes --collect_n=100000 &
#wait
#python train.py --env=Dropbox --datapath=rollouts/Dropbox-100000.npz --logdir=logs/paper/Dropbox/3_1024_32 --log_n=1 --save_n=1 --n_embed=1024 --n_head=32
python train.py --env=Bounce --datapath=rollouts/Bounce-100000.npz --logdir=logs/paper/Bounce/3_1024_32 --log_n=1 --save_n=1 --n_embed=1024 --n_head=32
python train.py --env=Urchin --datapath=rollouts/Urchin-100000.npz --logdir=logs/paper/Urchin/3_1024_32 --log_n=1 --save_n=1 --n_embed=1024 --n_head=32
python train.py --env=UrchinBall --datapath=rollouts/UrchinBall-100000.npz --logdir=logs/paper/UrchinBall/3_1024_32 --log_n=1 --save_n=1 --n_embed=1024 --n_head=32
python train.py --env=UrchinBalls --datapath=rollouts/UrchinBalls-100000.npz --logdir=logs/paper/UrchinBalls/3_1024_32 --log_n=1 --save_n=1 --n_embed=1024 --n_head=32
python train.py --env=UrchinCubes --datapath=rollouts/UrchinCubes-100000.npz --logdir=logs/paper/UrchinCubes/3_1024_32 --log_n=1 --save_n=1 --n_embed=1024 --n_head=32

