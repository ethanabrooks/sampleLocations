#! /usr/bin/env python

from functools import partial
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial import distance

PATH_LENGTH = 400
N_CHOICES = 40
ITERATIONS = 2


def caching(f):
  cache = {}

  """
  :param f: function that takes `*args` as argument
  :param args: list of args to `f
  :return: 
  """

  def try_cache_first(*args):
    hash_args = []
    for arg in args:
      if type(arg) is list:
        hash_args.append(tuple(arg))
      if type(arg) is np.ndarray:
        hash_args.append(arg.tostring())
      else:
        hash_args.append(arg)
    key = hash(tuple(hash_args))
    try:
      return cache[key]
    except KeyError:
      result = f(*args)
      cache[key] = result
      return result

  return try_cache_first


@caching
def get_cost_caching(path):
  """
  :param path: n x 2 np.array
  :return: the sum of mean squared errors from the first point in the path:
  >>>get_cost(np.array([(0, 0), (1, 0), (0, 1)]))
  2.0
  >>>get_cost(np.array([(0, 0), (0, 2), (0, -1)]))
  5.0
  """
  if path.size == 0:
    return 0
  else:
    head = path[0]
    tail = path[1:]
    return sum([distance.euclidean(head, pos) for pos in tail])


def get_cost(path):
  """
  :param path: n x 2 np.array
  :return: the sum of mean squared errors from the first point in the path:
  >>>get_cost(np.array([(0, 0), (1, 0), (0, 1)]))
  2.0
  >>>get_cost(np.array([(0, 0), (0, 2), (0, -1)]))
  5.0
  """
  if path.size == 0:
    return 0
  else:
    head = path[0]
    tail = path[1:]
    return sum([distance.euclidean(head, pos) for pos in tail])


def random_walk(time_steps):
  pos = np.zeros((time_steps, 2), dtype=float)
  vel = np.zeros(2, dtype=float)
  for i in range(1, time_steps):
    acc = np.random.normal(0, .001, 2)  # mean acceleration is 0
    vel += acc

    # bias toward rest
    # note that the larger the ratio of the second number to the first,
    # the more jagged the line. Also both numbers must be positive.
    # vel *= np.random.beta(1, 3)
    vel *= np.random.exponential(.5, ())
    pos[i] = pos[i - 1] + vel
  return pos

if __name__ == '__main__':
  split = np.split(random_walk(100), 2, axis=1)
  # plt.plot(n_choices, mse_uniform, '-o', color=color1, label='Uniform')
  plt.plot(*split, marker='o')
  plt.show()

