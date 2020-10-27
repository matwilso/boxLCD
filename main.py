from cfg import define_cfg, args_type, env_fn, make_env

from algo import sac, collect, dyn

if __name__ == '__main__':
    parser = define_cfg()
    parser.set_defaults(**{'exp_name': 'collect'})
    cfg = parser.parse_args()

    T = {'collect': collect.Collect, 'dyn': dyn.Dyn, 'sac': sac.SAC}[cfg.mode]
    t = T(cfg, make_env)
    t.run()
