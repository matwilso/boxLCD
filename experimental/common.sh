# COMMON SHELL COMMANDS THAT I SAVE SO I DON'T FORGET THEM

# collect data 
python3 main.py --mode=collect --env=dropbox --collect_n=10000
# train
python3 main.py --mode=world --env=dropbox --datapath=$dp --logdir=logs/train/


# convert a set of images to a single video gif
convert -resize 100% -delay 2 -loop 0 *.png test.gif