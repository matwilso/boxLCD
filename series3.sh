
# total time is 3hrs, so in seconds: 3*60*60 = 10800
# or 4hrs: 4*60*60 = 14400

# define common args
COMMON="python /home/matwilso/code/boxLCD/research/main.py --mode=train --model=video_rin --datadir=logs/datadump/Bounce2_large/ --save_n=1 --log_n=1000 --timesteps 100 --data_workers 4 --n_head 8 --total_time 10800"

# experiments to run
# - 32x16x16, patch size 4x4x4
# - 32x32x32, patch size 4x4x4
# - 32x32x32, patch size 4x8x8
# - 32x32x32, patch size 8x8x8

$COMMON --bs 50 --logdir logs/jan08/vid/exp4/01 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 32 --dim_z 512 --self_cond 0.5 --window 32 --dst_resolution 16 --patch_t 4 --patch_h 4 --patch_w 4

$COMMON --bs 50 --logdir logs/jan08/vid/exp4/02 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 32 --dim_z 512 --self_cond 0.5 --window 32 --dst_resolution 32 --patch_t 4 --patch_h 4 --patch_w 4

$COMMON --bs 50 --logdir logs/jan08/vid/exp4/03 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 32 --dim_z 512 --self_cond 0.5 --window 32 --dst_resolution 32 --patch_t 4 --patch_h 8 --patch_w 8

$COMMON --bs 50 --logdir logs/jan08/vid/exp4/04 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 32 --dim_z 512 --self_cond 0.5 --window 32 --dst_resolution 32 --patch_t 8 --patch_h 8 --patch_w 8

$COMMON --bs 50 --logdir logs/jan08/vid/exp4/06 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 32 --dim_z 512 --self_cond 0.5 --window 32 --dst_resolution 32 --patch_t 2 --patch_h 4 --patch_w 4 --total_time 21600

$COMMON --bs 50 --logdir logs/jan08/vid/exp4/07 --lr 1e-4 --num_layers 2 --num_blocks 3 --n_z 32 --dim_z 512 --self_cond 0.5 --window 32 --dst_resolution 32 --patch_t 2 --patch_h 2 --patch_w 2 --total_time 21600

$COMMON --bs 50 --logdir logs/jan08/vid/exp4/08 --lr 5e-4 --num_layers 2 --num_blocks 3 --n_z 32 --dim_z 512 --self_cond 0.5 --window 32 --dst_resolution 32 --patch_t 4 --patch_h 4 --patch_w 4 --total_time 21600