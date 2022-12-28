import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange, reduce, repeat

from boxLCD import env_map
from research import data, runners, utils
from research.define_config import env_fn, load_G
from research.nets import net_map

if __name__ == '__main__':
    G = load_G()

    # assert G.lcd_w == data_cfg.lcd_w and G.lcd_h == data_cfg.lcd_w, "mismatch of env dims"
    env = env_fn(G)()
    if G.mode not in ['collect', 'eval']:
        if G.model in net_map:
            model = net_map[G.model](env, G)
        else:
            assert False, f"we don't have that model, {G.model} {net_map}"
        model.to(G.device)
        G.num_vars = utils.count_vars(model)

    if G.mode == 'train':
        runner = runners.Trainer(model, env, G)
    elif G.mode == 'eval':
        runner = runners.Evaler(None, env, G)
    elif G.mode == 'viz':
        runner = runners.Vizer(model, env, G)
    elif G.mode == 'collect':
        data.collect(env_fn, G)
        exit()
    elif G.mode == 'eval_diffusion':
        with torch.no_grad():
            b = lambda x: {key: val.to(G.device) for key, val in x.items()}
            Net = net_map['diffusion_model']
            MODEL_BASE = (
                '/home/matwilso/code/boxLCD/logs/dec28/base_16_6/DiffusionModel.pt'
            )
            MODEL_32 = '/home/matwilso/code/boxLCD/logs/dec28/to32/DiffusionModel.pt'
            MODEL_64 = '/home/matwilso/code/boxLCD/logs/dec28/to64/DiffusionModel.pt'
            base = Net.from_disk(MODEL_BASE)
            to32 = Net.from_disk(MODEL_32)
            to64 = Net.from_disk(MODEL_64)

            arbiter_path = Path(
                '/home/matwilso/code/boxLCD/logs/dec28/arbiter8/ArbiterAE.pt'
            )
            arbiter = torch.jit.load(str(arbiter_path))
            with (arbiter_path.parent / 'hps.yaml').open('r') as f:
                arbiterG = yaml.load(f, Loader=yaml.Loader)
            arbiter.G = arbiterG
            arbiter.eval()

            G.window = 1
            N = 500
            G.bs = N // G.window
            train_ds, test_ds = data.load_ds(G, resolutions=[16, 32, 64])
            flat = lambda x: rearrange(x, 'b c t h w -> (b t) c h w')

            def show(name, x):
                x = rearrange(
                    F.pad(x[:25], (0, 1, 0, 1)),
                    '(n1 n2) c h w -> (n1 h) (n2 w) c',
                    n1=5,
                    n2=5,
                )
                x = (x + 1) / 2
                plt.imsave(f'./logs/{name}.png', x.detach().cpu().numpy())

            for batch in test_ds:
                batch = b(batch)
                print("BASE")
                sample16 = base.sample(N)
                print("32")
                sample32 = to32.sample(
                    N, low_res=Net.upres_coarse(batch['lcd_32'], sample16['lcd_16'])
                )
                print("64")
                sample64 = to64.sample(
                    N, low_res=Net.upres_coarse(batch['lcd_64'], sample32['lcd_32'])
                )
                genx = sample64['lcd_64']
                datax = flat(batch['lcd'])
                proprio = rearrange(batch['proprio'], 'b t c -> (b t) c')
                dataz = arbiter({'lcd': datax, 'proprio': proprio})
                genz = arbiter({'lcd': genx, 'proprio': proprio})
                fid = utils.compute_fid(
                    dataz.detach().cpu().numpy(), genz.detach().cpu().numpy()
                )
                print("FID", fid)

                show('data', datax)
                show('gen', genx)

                # cheats
                cheat_sample32 = to32.sample(
                    N, low_res=Net.upres_coarse(batch['lcd_32'], flat(batch['lcd_16']))
                )
                cheat_sample64 = to64.sample(
                    N, low_res=Net.upres_coarse(batch['lcd_64'], flat(batch['lcd_32']))
                )

                upto64 = lambda x: Net.upres_coarse(batch['lcd_64'], x)
                batch16_up = upto64(flat(batch['lcd_16']))
                batch32_up = upto64(flat(batch['lcd_32']))

                # we need to figure out where we go wrong.

                # so we can compare base sample to batch16_up
                # and we can compare sample32 to batch32_up
                base16_up = upto64(sample16['lcd_16'])
                base32_up = upto64(sample32['lcd_32'])

                show('base16_up', base16_up)
                show('base32_up', base32_up)
                show('batch16_up', batch16_up)
                show('batch32_up', batch32_up)

                dataz = arbiter({'lcd': batch16_up, 'proprio': proprio})
                genz = arbiter({'lcd': base16_up, 'proprio': proprio})
                fid = utils.compute_fid(
                    dataz.detach().cpu().numpy(), genz.detach().cpu().numpy()
                )
                print("base16_up FID", fid)

                dataz = arbiter({'lcd': batch32_up, 'proprio': proprio})
                genz = arbiter({'lcd': base32_up, 'proprio': proprio})
                fid = utils.compute_fid(
                    dataz.detach().cpu().numpy(), genz.detach().cpu().numpy()
                )
                print("base32_up FID", fid)
                breakpoint()

    if G.ipython_mode:
        import IPython
        from traitlets.config import Config

        c = Config()
        c.InteractiveShellApp.exec_lines = ['runner.run()']
        c.TerminalInteractiveShell.banner2 = '***Welcome to Quick Iter Mode***'
        IPython.start_ipython(config=c, user_ns=locals())
    else:
        runner.run()
