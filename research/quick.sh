#python3 scripts/kicker.py arbiter --model=MultiStepArbiter --total_itr=1000 --log_n=100
#python3 scripts/kicker.py train --model=BVAE --total_itr=1000 --log_n=100
#python3 scripts/kicker.py train --model=RNLDA --total_itr=1000 --log_n=100
#python3 scripts/kicker.py train --model=RSSM --total_itr=1000 --log_n=100
python3 scripts/kicker.py train --model=FIT --total_itr=1000 --log_n=100
python3 scripts/kicker.py train --model=FBT --total_itr=1000 --log_n=100
python3 scripts/kicker.py train --model=FRNLD --total_itr=1000 --log_n=100