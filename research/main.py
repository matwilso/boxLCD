import argparse

import yaml

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
        Net = net_map['diffusion_model']
        MODEL_BASE = '/home/matwilso/code/boxLCD/logs/dec28/base_16_1e-3/net.pt'
        #MODEL_32 = ''
        #MODEL_64 = ''
        base = Net.from_disk(MODEL_BASE, device='cuda')
        b = lambda x: {key: val.to(G.device) for key, val in x.items()}
        #to32 = Net.from_disk(MODEL_32)
        #to64 = Net.from_disk(MODEL_64)
        train_ds, test_ds = data.load_ds(G, resolutions=[16, 32, 64])
        for batch in test_ds:
            batch = b(batch)
            n = batch['lcd'].shape[0]
            sample16 = base.sample(n)
            #sample32 = to32.sample(n, low_res=sample16['lcd_16'])
            #sample64 = to64.sample(n, low_res=sample32['lcd_32'])
            # cheats
            #cheat_sample32 = to32.sample(n, low_res=batch['lcd_16'])
            #cheat_sample64 = to64.sample(n, low_res=batch['lcd_32'])



    if G.ipython_mode:
        import IPython
        from traitlets.config import Config

        c = Config()
        c.InteractiveShellApp.exec_lines = ['runner.run()']
        c.TerminalInteractiveShell.banner2 = '***Welcome to Quick Iter Mode***'
        IPython.start_ipython(config=c, user_ns=locals())
    else:
        runner.run()
