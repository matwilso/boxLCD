import time
from boxLCD.envs import Bounce

env = Bounce()
while True:
    start = time.time()
    env.reset()
    print(time.time()-start)
