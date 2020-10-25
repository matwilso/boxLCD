from cfg import define_cfg, args_type, env_fn, make_env
#from algo.sac import SAC
from algo.collect import Collect

if __name__ == '__main__':
    parser = define_cfg()
    parser.set_defaults(**{'exp_name': 'collect'})
    cfg = parser.parse_args()
    t = Collect(cfg, make_env)
    t.run()