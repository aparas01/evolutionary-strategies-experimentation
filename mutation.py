import numpy as np
from itertools import repeat
import random
import math

from model import Individual


class Mutation:
    def mutate(self, individual: Individual) -> Individual:
        pass

    def setup(self, individual_size):
        pass

    def has_alphas(self):
        pass

class SingleSigma(Mutation):
    def __init__(self, pm: float):
        self.pm = pm
        self.mu = 0

    def setup(self, individual_size):
        self.t0 = 1/math.sqrt(2*individual_size)

    def has_alphas(self):
        return False

    def mutate(self, individual: Individual) -> Individual:
        '''
            This function is used to apply Guassian (normal distribution) mutation 
            on an individual
            param self: self referencing to access attributes such as learning rate (t0), mu, sigma and pm
            param individual: The individual to be mutated
        '''
        size = individual.size()
        new_genotype = individual.get_genotype()
        new_sigmas = individual.get_sigmas() * math.exp(random.gauss(self.mu, self.t0))

        for i, s in zip(range(size), new_sigmas):
            if random.random() < self.pm:
                new_genotype[i] += random.gauss(self.mu, s)
        new_individual = Individual(new_genotype, sigmas=new_sigmas, alphas=[])
        return new_individual


class IndividualSigmaPerGene(Mutation):
    def __init__(self, pm: float):
        self.pm = pm
        self.mu = 0

    def setup(self, individual_size):
        self.t0 = 1/math.sqrt(2*math.sqrt(individual_size))
        self.t0_prime = 1/math.sqrt(2*individual_size)

    def has_alphas(self):
        return False

    def mutate(self, individual: Individual) -> Individual:
        '''
            This function is used to apply Guassian (normal distribution) mutation 
            on an individual
            param self: self referencing to access attributes such as learning rate (t0), mu, sigma and pm
            param individual: The individual to be mutated
        '''
        size = individual.size()
        new_genotype = individual.get_genotype()
        new_sigmas = individual.get_sigmas()
        prime_gaussian = random.gauss(self.mu, self.t0_prime) 
        for index, _ in enumerate(new_sigmas):
            new_sigmas[index] = new_sigmas[index] * math.exp(
                prime_gaussian
                + random.gauss(self.mu, self.t0)
            )
        for i, s in zip(range(size), new_sigmas):
            if random.random() < self.pm:
                new_genotype[i] += random.gauss(self.mu, s)
        new_individual = Individual(new_genotype, sigmas=new_sigmas, alphas=[])
        return new_individual


class CorrelatedSigma(Mutation):
    def __init__(self, pm: float):
        self.pm = pm
        self.mu = 0

    def setup(self, individual_size):
        self.t0 = 1/math.sqrt(2*individual_size)
        self.t0_prime = 1/math.sqrt(2*math.sqrt(individual_size))
        self.beta = math.pi/36

    def has_alphas(self):
        return True

    def mutate(self, individual: Individual) -> Individual:
        '''
            This function is used to apply Guassian (normal distribution) mutation 
            on an individual
            param self: self referencing to access attributes such as learning rate (t0), mu, sigma and pm
            param individual: The individual to be mutated
        '''
        size = individual.size()
        final_r = np.identity(size)
        count = 0
        new_genotype = individual.get_genotype()
        old_sigmas = individual.get_sigmas()
        new_alphas = individual.get_alphas()
        dash_gaussian = random.gauss(self.mu, self.t0_prime) 
        for index, _ in enumerate(old_sigmas):
            old_sigmas[index] = old_sigmas[index] * math.exp(
                dash_gaussian
                + random.gauss(self.mu, self.t0)
            )
        for index, _ in enumerate(new_alphas):
            new_alpha = new_alphas[index] + random.gauss(self.mu, self.beta)
            if abs(new_alpha) > math.pi:
                new_alpha = new_alpha - 2*math.pi * np.sign(new_alpha)
            new_alphas[index] = new_alpha
        s = np.zeros((size, size), dtype='float64')

        for i in range(0, size):
            for j in range(i+1, size):
                r = np.identity(size)
                r[i][i] = math.cos(new_alphas[count])
                r[j][j] = math.cos(new_alphas[count])
                r[j][i] = math.sin(new_alphas[count])
                r[i][j] = -math.sin(new_alphas[count])
                final_r = np.dot(final_r, r)
                count+=1
        np.fill_diagonal(s, old_sigmas)   

        C = np.dot(final_r, s)
        C = np.dot(C, C.T)
    
        mu = np.full((individual.size()), fill_value=self.mu)
        new_sigmas = np.random.multivariate_normal(mean=mu, cov=C)

        for i, s in zip(range(size), new_sigmas):
            new_genotype[i] += random.gauss(self.mu, s)

        new_individual = Individual(new_genotype, sigmas=old_sigmas, alphas=new_alphas)
        return new_individual