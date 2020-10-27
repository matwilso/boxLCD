import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from algo.base import Trainer
from tensorflow import nest
from torch import distributions
def fvmap(f, dim):
    """fakes vmap by just reshaping"""
    def _thunk(x):
        preshape = x.shape
        x = f(x.flatten(0, 1))
        import ipdb; ipdb.set_trace()
        return x.reshape

class ConvEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._depth = 32
        self.c1 = nn.Conv2d(3, self._depth, kernel_size=4, stride=2)
        self.c2 = nn.Conv2d(self._depth, 2*self._depth, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(2*self._depth, 4*self._depth, kernel_size=4, stride=2)
        self.c4 = nn.Conv2d(4*self._depth, 8*self._depth, kernel_size=4, stride=2)

    def forward(self, obs):
        x = self.c1(obs['image'])
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.c4(x)
        return x.flatten(1, -1)

class ConvDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._depth = 32
        self._shape = (64, 64, 3)
        #self.l1 = nn.Linear(1024, 32*self._depth)
        self.d1 = nn.ConvTranspose2d(32*self._depth, 4*self._depth, 5, stride=2)
        self.d2 = nn.ConvTranspose2d(4*self._depth, 2*self._depth, 5, stride=2)
        self.d3 = nn.ConvTranspose2d(2*self._depth, 1*self._depth, 6, stride=2)
        self.d4 = nn.ConvTranspose2d(1*self._depth, 3, 6, stride=2)

    def forward(self, features):
        x = features[...,None,None]
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = F.relu(x)
        x = self.d3(x)
        x = F.relu(x)
        x = self.d4(x)
        #return F.sigmoid(x)
        return distributions.Independent(distributions.Bernoulli(logits=x), len(self._shape))
        #return distributions.Independent(distributions.Normal(x,1), len(self._shape))

class Dyn(Trainer):
    def __init__(self, cfg, make_env):
        super().__init__(cfg, make_env)
        self.encoder = ConvEncoder(cfg).to(cfg.device)
        self.decoder = ConvDecoder(cfg).to(cfg.device)
        self.params = itertools.chain(self.encoder.parameters(), self.decoder.parameters())
        self.optimizer = optim.Adam(self.params, lr=self.cfg.dyn_lr)
        # TODO: try out just autoencoder first

    def loss(self):
        pass
        
    def update(self, batch):
        for i in range(len(batch['act'][0])):
            sb = nest.map_structure(lambda x: x[i], batch)
            self.optimizer.zero_grad()
            embed = self.encoder(sb)
            decode = self.decoder(embed)
            #loss = 0.5*(decode - sb['image'])**2
            #loss = loss.mean()
            loss = -decode.log_prob(sb['image']).mean()
            loss.backward()
            self.optimizer.step()
            self.logger['Loss'] += [loss.detach().cpu()]
            info = {}
            info['recon'] = (decode.mean[:6] + 0.5).detach().cpu()
        return info

    def run(self):
        self.refresh_dataset()
        start_time = time.time()
        epoch_time = time.time()
        for t in itertools.count(1):
            bt = time.time()
            batch = next(self.data_iter)
            batch = nest.map_structure(lambda x: torch.tensor(x).to(self.cfg.device), batch)
            batch['image'] = batch['image'].permute([0, 1, -1, 2, 3])/ 255.0
            #batch = nest.map_structure(lambda x: x.flatten(0,1), batch)
            self.logger['bdt'] += [time.time() - bt]

            ut = time.time()
            info = self.update(batch)
            self.logger['udt'] += [time.time() - ut]
            if t % 100 == 0:
                print('='*30)
                print('t', t)
                for key in self.logger:
                    x = np.mean(self.logger[key])
                    self.writer.add_scalar(key, x, t)
                    print(key, x)
                self.writer.add_images('recon', info['recon'], t)
                self.writer.flush()
                print('dt', time.time()-epoch_time)
                print('total time', time.time()-start_time)
                print(self.logpath)
                print(self.cfg.full_cmd)
                print('='*30)
                epoch_time = time.time()

