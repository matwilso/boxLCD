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

