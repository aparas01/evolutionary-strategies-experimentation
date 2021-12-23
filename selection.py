import numpy as np
from model import Population, Pair
from typing import List


class Selection:
    def select_pairs(self, population: Population, pairs_size: int) -> List[Pair]:
        pass


class RandomSelection(Selection):

    def info(self):
        return f'random'

    def select_pairs(self, population: Population, pairs_size: int) -> List[Pair]:
        pairs = []
        for _ in range(pairs_size):
            o1, o2 = np.random.choice(population.individuals, size=2)
            pairs.append(Pair(o1, o2))

        return pairs
