import yaml
from datetime import datetime
import argparse
from hps import hps, args_type
import runners

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for key, value in hps().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  H = parser.parse_args()