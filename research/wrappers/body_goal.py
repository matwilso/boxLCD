import copy

import numpy as np
from PIL import Image

from research import utils


class BodyGoalEnv:
    def __init__(self, env, G):
        self._env = env
        self.SCALE = 2
        self.G = G

    def seed(self, *args):
        self._env.seed(*args)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        base_space = self._env.observation_space
        base_space.spaces['goal:lcd'] = base_space.spaces['lcd']
        base_space.spaces['goal:proprio'] = base_space.spaces['proprio']
        return base_space

    def reset(self, *args, **kwargs):
        self.goal = self._env.reset()
        obs = self._env.reset(*args, **kwargs)
        # self.goal = obs = self._env.reset(*args, **kwargs)
        obs['goal:lcd'] = np.array(self.goal['lcd'])
        obs['goal:proprio'] = np.array(self.goal['proprio'])
        self.last_obs = copy.deepcopy(obs)
        return obs

    def simi2rew(self, similarity):
        """map [0,1] --> [-1,0] in an exponential mapping. (if you get linearly closer to 1.0, you get exponentially closer to 0.0)"""
        assert similarity >= 0.0 and similarity <= 1.0
        # return np.exp(SCALE*similarity) / np.exp(SCALE*1)
        return -1 + similarity
        # return -1 + np.exp(self.SCALE * (similarity - 1))

    # def rew2simi(self, rew):
    #  """map [-1,0] --> [-1,0] in a log mapping."""
    #  assert rew >= -1.0 and rew <= 0.0
    #  return (np.log(rew + 1) / self.SCALE) + 1
    def render(self, *args, **kwargs):
        self._env.render(*args, **kwargs)

    def comp_rew_done(self, obs, info={}):
        done = False
        if self.G.state_rew:
            delta = np.abs(obs['goal:proprio'] - obs['proprio'])
            # keys = utils.filtlist(self._env.pobs_keys, '.*x:p')
            keys = utils.filtlist(self._env.pobs_keys, '.*(x|y):p')
            idxs = [self._env.pobs_keys.index(x) for x in keys]
            delta = delta[idxs].mean()
            if self.G.diff_delt:
                last_delta = np.abs(
                    self.last_obs['goal:proprio'] - self.last_obs['proprio']
                )
                last_delta = last_delta[idxs].mean()
                rew = -0.05 + 10 * (last_delta - delta)
            else:
                rew = -delta

            info['delta'] = delta
            if delta < self.G.goal_thresh:
                rew += 1.0
                # done = False
                info['success'] = True
                done = True
        else:
            similarity = (
                np.logical_and(obs['lcd'] == 0, obs['lcd'] == obs['goal:lcd']).mean()
                / (obs['lcd'] == 0).mean()
            )
            rew = -1 + similarity
            info['delta'] = similarity
            if similarity > 0.70:
                rew = 0
                info['success'] = True
                # done = False
                done = True
        return rew, done

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        obs['goal:lcd'] = np.array(self.goal['lcd'])
        obs['goal:proprio'] = np.array(self.goal['proprio'])
        rew, _done = self.comp_rew_done(obs, info)
        done = done or _done
        # similarity = (obs['goal:lcd'] == obs['lcd']).mean()
        # rew = self.simi2rew(similarity)
        rew = rew * self.G.rew_scale
        self.last_obs = copy.deepcopy(obs)
        return obs, rew, done, info

    def close(self):
        self._env.close()


if __name__ == '__main__':
    pass

    import utils
    from PIL import Image, ImageDraw, ImageFont
    from rl.sacnets import ActorCritic

    from boxLCD import envs

    G = utils.AttrDict()
    G.state_rew = 1
    G.device = 'cpu'
    G.lcd_h = 16
    G.lcd_w = 32
    G.wh_ratio = 2.0
    G.lr = 1e-3
    # G.lcd_base = 32
    G.rew_scale = 1.0
    G.diff_delt = 1
    G.env = 'Luxo'
    env = envs.Luxo(G)
    # G.env = 'Urchin'
    # env = envs.Urchin(G)
    G.fps = env.G.fps
    env = BodyGoalEnv(env, G)
    print(env.observation_space, env.action_space)
    obs = env.reset()
    lcds = [obs['lcd']]
    glcds = [obs['goal:lcd']]
    rews = [-np.inf]
    deltas = [-np.inf]
    while True:
        env.render(mode='human')
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)
        # o = {key: torch.as_tensor(val[None].astype(np.float32), dtype=torch.float32).to(G.device) for key, val in obs.items()}
        lcds += [obs['lcd']]
        glcds += [obs['goal:lcd']]
        rews += [rew]
        deltas += [info['delta']]
        # plt.imshow(obs['lcd'] != obs['goal:lcd']); plt.show()
        # plt.imshow(np.c_[obs['lcd'], obs['goal:lcd']]); plt.show()
        if done:
            break

    def outproc(img):
        return (
            (255 * img[..., None].repeat(3, -1))
            .astype(np.uint8)
            .repeat(8, 1)
            .repeat(8, 2)
        )

    lcds = np.stack(lcds)
    glcds = np.stack(glcds)
    lcds = (1.0 * lcds - 1.0 * glcds + 1.0) / 2.0
    lcds = outproc(lcds)
    dframes = []
    for i in range(len(lcds)):
        frame = lcds[i]
        pframe = Image.fromarray(frame)
        # get a drawing context
        draw = ImageDraw.Draw(pframe)
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 60)
        color = (255, 255, 255)
        # draw.text((10, 10), f't: {i} r: {rews[i]:.3f}\nd: {deltas[i]:.3f}', fnt=fnt)
        draw.text(
            (10, 10),
            f't: {i} r: {rews[i]:.3f}\nd: {deltas[i]:.3f}',
            fill=color,
            fnt=fnt,
        )
        dframes += [np.array(pframe)]
    dframes = np.stack(dframes)
    utils.write_video('mtest.mp4', dframes, fps=G.fps)
