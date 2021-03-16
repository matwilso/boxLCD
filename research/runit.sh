#python main.py --mode=train --env=UrchinBall --datapath=logs/datadump/urchinball/update/1e4/dump.npz --model=multistep --device=cuda --weightdir=logs/biphase/x2/ --vidstack=4 --phase=2 --log_n=1 --logdir=logs/biphase/x2/phase2/nl3_512_16/1e-3 --n_layer=3 --n_embed=512 --n_head=16 --bs=16 --amp=0 --lr=1e-3 --num_epochs=5

python main.py --mode=train --env=UrchinBall --datapath=logs/datadump/urchinball/update/1e4/dump.npz --model=multistep --device=cuda --weightdir=logs/biphase/x2/ --vidstack=4 --phase=2 --log_n=1 --logdir=logs/biphase/x2/phase2/fewer_stack/nl3_512_16/ --n_layer=3 --n_embed=512 --n_head=16 --bs=16 --amp=0 --stacks_per_block=16 --num_epochs=5


python main.py --mode=train --env=UrchinBall --datapath=logs/datadump/urchinball/update/1e4/dump.npz --model=multistep --device=cuda --weightdir=logs/biphase/x2/ --vidstack=4 --phase=2 --log_n=1 --logdir=logs/biphase/x2/phase2/fewer_stack/nl3_1024_32/ --n_layer=3 --n_embed=1024 --n_head=32 --bs=16 --amp=0 --stacks_per_block=16 --num_epochs=5

python main.py --mode=train --env=UrchinBall --datapath=logs/datadump/urchinball/update/1e4/dump.npz --model=multistep --device=cuda --weightdir=logs/biphase/x2/ --vidstack=4 --phase=2 --log_n=1 --logdir=logs/biphase/x2/phase2/fewer_stack/nl4_1024_32/ --n_layer=4 --n_embed=1024 --n_head=32 --bs=16 --amp=0 --stacks_per_block=16 --num_epochs=5

python main.py --mode=train --env=UrchinBall --datapath=logs/datadump/urchinball/update/1e4/dump.npz --model=multistep --device=cuda --weightdir=logs/biphase/x2/ --vidstack=4 --phase=2 --log_n=1 --logdir=logs/biphase/x2/phase2/fewer_stack/nl3_512_16/bs32 --n_layer=3 --n_embed=512 --n_head=16 --bs=32 --amp=0 --stacks_per_block=16 --num_epochs=5


