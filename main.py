from cfg import define_cfg, args_type, env_fn, make_env
from algo import sac, collect, dyn, viz

if __name__ == '__main__':
    parser = define_cfg()
    cfg = parser.parse_args()

    T = {'collect': collect.Collect, 'dream': dyn.Dyn, 'dyn': dyn.Dyn, 'sac': sac.SAC, 'viz': viz.Viz}[cfg.mode]
    t = T(cfg, make_env)
    t.run()
