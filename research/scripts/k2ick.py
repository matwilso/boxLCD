#!/usr/bin/env python3
from collections import defaultdict
import click
import subprocess

default_color = "magenta"
acc_color = "yellow"

class StdCommand(click.Command):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.params += [click.Option(('--tier', '-t'), default=0)]
    self.params += [click.Option(('--dry_run','-d'), is_flag=True, help="Show what commands are going to be run, but don't run them")]
    self.script_name = 'research.main'

class TrainCommand(StdCommand):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.params += [click.Option(('--tier', '-t'), default=0)]
    self.params += [click.Option(('--dry_run','-d'), is_flag=True, help="Show what commands are going to be run, but don't run them")]


@click.group()
def main():
  """
  Kick off a group of tests
  """
  pass

@main.command(cls=StdCommand)
#@click.option('--dry_run', '-d', is_flag=True, help="Show what commands are going to be run, but don't run them")
@click.pass_context
def collect(ctx, **kwargs):
  click.echo(kwargs)
  command = [
    'python3 -m',
    ctx.command.script_name,
    '--num_envs=10',
    '--train_barrels=100',
    '--test_barrels=10',
  ]
  import ipdb; ipdb.set_trace()

    #python -m research.main --mode=collect --num_envs=10 --train_barrels=100 --test_barrels=10 --env=$env --logdir=logs/datadump/$env
  #click.secho(f'TIER: {kwargs["tier"]}', fg=default_color)
  #subprocess.run(["ls", "-l", "/dev/null"], capture_output=True)
  pass

@main.command(cls=TrainCommand)
@click.pass_context
def arbiter(**kwargs):
  pass

@main.command(cls=TrainCommand)
@click.pass_context
def bvae(**kwargs):
  pass


if __name__ == '__main__':
  main()
