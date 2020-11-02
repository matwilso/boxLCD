import time
import torch
import torch.nn as nn
import jax
import trax
import trax.layers as tl
import jax.numpy as jnp
F = nn.functional

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, 128)
    def forward(self, obs):
        x = obs
        x = self.l1(x)
        x = F.relu(x)

        x = self.l2(x)
        x = F.relu(x)

        x = self.l3(x)
        x = F.relu(x)

        x = self.l4(x)
        return x


if True:
    models = [NN().cuda() for _ in range(1)]
    obs = torch.rand(1024, 128).cuda()
    outs = []
    for m in models:
        outs.append(m(obs))
    out = torch.stack(outs)

    start = time.time()
    outs = []
    for m in models:
        outs.append(m(obs))
    out = torch.stack(outs)
    print(out, out.shape)
else:
    rng = jax.random.PRNGKey(0)
    rng1 = jax.random.PRNGKey(1)
    obs = jax.random.normal(rng, (1024, 128))
    obs2 = jax.random.normal(rng1, (1024, 128))
    model = tl.Serial(
        tl.Dense(128),
        tl.Relu(),
        tl.Dense(128),
        tl.Relu(),
        tl.Dense(128),
        tl.Relu(),
        tl.Dense(128),
    )
    model.init(trax.shapes.signature(obs))
    weights = jax.tree_util.tree_map(lambda x: jnp.tile(x, [5, *([1]*len(x.shape))]), model.weights)
    fwd = jax.jit(jax.vmap(model.pure_fn, (None,0,None,None)))
    fwd(obs, weights, model.state, None)
    start = time.time()
    out = fwd(obs2, weights, model.state, None)[0]
    print(out, out.shape)

print(time.time() - start)
