from boxLCD import envs, C, wrappers
env = envs.UrchinBall(C)
#env = wrappers.LCDEnv(env)  # give lcd images as part of the observation
obs = env.reset()
while True:
    action = env.action_space.sample()
    obs, _, done, info = env.step(action)
    env.render(mode='human')
    # get the numpy arrays corresponding to lcd and pretty rendering
    #lcd = env.lcd_render(pretty=False)  
    #pretty = env.lcd_render(pretty=True)