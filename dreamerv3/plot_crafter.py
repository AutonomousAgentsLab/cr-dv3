"""Simple script to plot results of a crafter run with dreamerv3-cr."""
import pathlib, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import json
import warnings
import argparse

def main():
  runs = []

  home = os.path.expanduser("~")
  default_filename = f'{home}/logdir/crafter-dv3-cr_1/stats.jsonl'

  parser = argparse.ArgumentParser(description='Process some arguments')
  parser.add_argument('--filename', default=default_filename,
                    help='The path to stats.jsonl file')
  args = parser.parse_args()

  filename = args.filename
  print(f"Filename: {filename}")
  budget=1e6
  rewards, lengths, achievements = load_stats(pathlib.Path(filename), budget)
  task, method, seed = pathlib.Path(filename).parts[-5:-2]

  print(f'Run length {sum(lengths)}: {filename}')
  runs.append(dict(
    task=task,
    method=method,
    seed=str(id),
    xs=np.cumsum(lengths).tolist(),
    reward=rewards,
    length=lengths,
    **achievements,
  ))

  scores, tasks, percents, methods = print_summary(runs, budget, verbose=True)

  if len(rewards) < 100:
    print('>>> Not plotting reward curve until at least 100 episodes saved out')
  else:
    plt.plot(np.cumsum(lengths).tolist(), pd.Series(rewards).rolling(window=100).mean().values)
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.show()

def load_stats(filename, budget):
  steps = 0
  rewards = []
  lengths = []
  achievements = collections.defaultdict(list)
  for line in filename.read_text().split('\n'):
    if not line.strip():
      continue
    episode = json.loads(line)
    steps += episode['length']
    if steps > budget:
      break
    lengths.append(episode['length'])
    for key, value in episode.items():
      if key.startswith('achievement_'):
        achievements[key].append(value)
    unlocks = int(np.sum([(v[-1] >= 1) for v in achievements.values()]))
    health = -0.9
    rewards.append(unlocks + health)
  return rewards, lengths, achievements


def print_summary(runs, budget, verbose):
  episodes = np.array([len(x['length']) for x in runs])
  rewards = np.array([np.mean(x['reward']) for x in runs])
  lengths = np.array([np.mean(x['length']) for x in runs])
  percents, methods, seeds, tasks = compute_success_rates(
      runs, budget, sortby=0)
  scores = np.squeeze(compute_scores(percents))
  print(f'Score:        {np.mean(scores):10.2f} ± {np.std(scores):.2f}')
  print(f'Reward:       {np.mean(rewards):10.2f} ± {np.std(rewards):.2f}')
  print(f'Length:       {np.mean(lengths):10.2f} ± {np.std(lengths):.2f}')
  print(f'Episodes:     {np.mean(episodes):10.2f} ± {np.std(episodes):.2f}')
  if verbose:
    for task, percent in zip(tasks, np.squeeze(percents).T):
      name = task[len('achievement_'):].replace('_', ' ').title()
      print(f'{name:<20}  {np.mean(percent):6.2f}%')
  return scores, tasks, percents, methods


def compute_success_rates(runs, budget=1e6, sortby=None):
  methods = sorted(set(run['method'] for run in runs))
  seeds = sorted(set(run['seed'] for run in runs))
  tasks = sorted(key for key in runs[0] if key.startswith('achievement_'))
  percents = np.empty((len(methods), len(seeds), len(tasks)))
  percents[:] = np.nan
  for run in runs:
    episodes = (np.array(run['xs']) <= budget).sum()
    i = methods.index(run['method'])
    j = seeds.index(run['seed'])
    for key, values in run.items():
      if key in tasks:
        k = tasks.index(key)
        percent = 100 * (np.array(values[:episodes]) >= 1).mean()
        if np.isnan(percent):
          print(percent)
        percents[i, j, k] = percent
  if isinstance(sortby, (str, int)):
    if isinstance(sortby, str):
      sortby = methods.index(sortby)
    order = np.argsort(-np.nanmean(percents[sortby], 0), -1)
    percents = percents[:, :, order]
    tasks = np.array(tasks)[order].tolist()
  return percents, methods, seeds, tasks


def compute_scores(percents):
  # Geometric mean with an offset of 1%.
  assert (0 <= percents).all() and (percents <= 100).all()
  if (percents <= 1.0).all():
    print('Warning: The input may not be in the right range.')
  with warnings.catch_warnings():  # Empty seeds become NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
  return scores

if __name__ == "__main__":
  main()
