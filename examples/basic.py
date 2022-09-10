import time

from boxLCD import envs

env = envs.UrchinBall()
obs = env.reset()
while True:
    action = env.action_space.sample()
    obs, _, done, info = env.step(action)
    env.render(mode='human')
    print(obs['lcd'] * 1.0, '\n')
    time.sleep(0.01)
