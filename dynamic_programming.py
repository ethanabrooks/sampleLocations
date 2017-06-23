#! /usr/bin/env python

from util import get_cost, random_walk, PATH_LENGTH, N_CHOICES, \
  caching


def best_choice_no_cache(path, n_queries):
  assert n_queries >= 0
  if path.size == 0:
    return [], 0
  if n_queries == 0:
    return [], get_cost(path)

  else:
    def cost_at(i):
      choices, cost = best_choice_no_cache(path[i:], n_queries - 1)
      error = get_cost(path[:i])
      return [i] + [i + j for j in choices], error + cost

    return min(list(map(cost_at, range(path.shape[0]))),
               key=lambda values: values[1])


@caching
def best_choice(path, n_samples):
  """
  cache all results of calls to this function and check cache 
  before calling this function.
  """
  if path.size == 0:
    return [], 0
  if n_samples == 0:
    return [], caching(get_cost)(path)

  else:
    def cost_at(i):
      """
      :param i: next index to choose
      :return: cost of choosing this index
      """
      choices, cost = best_choice(path[i:], n_samples - 1)
      error = caching(get_cost)(path[:i])
      return [i] + [i + j for j in choices], error + cost

    return min(list(map(cost_at, range(path.shape[0]))),
               key=lambda values: values[1])

if __name__ == '__main__':
    path = random_walk(PATH_LENGTH)
    choices = best_choice_no_cache(path, N_CHOICES)
    print(choices)
    choices = best_choice(path, N_CHOICES)
    print(choices)


