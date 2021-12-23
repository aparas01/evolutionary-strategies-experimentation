from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Individual:
    genotype: np.ndarray
    sigmas: np.ndarray
    alphas: np.ndarray
    fitness: float = None

    def size(self):
        return len(self.genotype)

    def copy_genotype(self):
        return Individual(self.genotype.copy())

    def get_genotype(self):
        return self.genotype.copy()
    
    def get_sigmas(self):
        return self.sigmas.copy()

    def get_alphas(self):
        return self.alphas.copy()

    # setter method
    def set_genotype(self, genotype):
        self.genotype = genotype

@dataclass
class Population:
  individuals: List[Individual]

  def get_fitnesses(self):
    return np.array(list(
      map(lambda i: i.fitness, self.individuals)
    ), dtype=float)

  def ave_fitness(self):
    return sum(individual.fitness for individual in self.individuals) / len(self.individuals)

  def average_individual(self):
    ave_fitness = self.ave_fitness()
    for i in self.individuals:
      if i.fitness == ave_fitness:
        return i

  def max_fitness(self):
    return max(individual.fitness for individual in self.individuals)

  def best_individual(self):
    min_fitness = self.min_fitness()
    for i in self.individuals:
      if i.fitness == min_fitness:
        return i

  def min_fitness(self):
    return min(individual.fitness for individual in self.individuals)

  def evaluate_fitnesses(self, func, budget):
    for individual in self.individuals:
      fitness, budget = evaluate_fitness(func, individual.genotype, budget)
      individual.fitness = fitness
    return budget

  def size(self):
    return len(self.individuals)

@dataclass
class Pair:
  parent1: Individual
  parent2: Individual

def evaluate_fitness(func, x, budget):
  return func(x), budget - 1
