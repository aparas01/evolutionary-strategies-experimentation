import numpy as np
from model import Individual, Population
from typing import List
from selection import Selection


class Recombination:
    def recombine(self, population: Population) -> List[Individual]:
        pass


class Discrete(Recombination):
    def __init__(self, offsprings_size: int, selection: Selection):
        self.offsprings_size = offsprings_size
        self.selection = selection

    def recombine(self, population: Population) -> List[Individual]:
        offspring_individuals = []

        pairs = self.selection.select_pairs(population, self.offsprings_size)
        for pair in pairs:
            
            genotype1 = pair.parent1.get_genotype()
            genotype2 = pair.parent2.get_genotype()
            choice = np.random.randint(2, size=len(genotype1)).astype(bool)
            size = len(genotype1)
            alpha_len = int(size*(size-1)/2)
            alpha_choice = np.random.randint(2, size=alpha_len).astype(bool)
            genotype = np.where(choice, genotype1, genotype2)
            sigmas = np.array([])
            alphas = np.array([])
            
            if (len(pair.parent1.get_sigmas()) != 0):
                sigmas1 = pair.parent1.get_sigmas()
                sigmas2 = pair.parent2.get_sigmas()
                sigmas = np.where(choice, sigmas1, sigmas2)

            if (len(pair.parent1.get_alphas()) != 0):
                alphas1 = pair.parent1.get_alphas()
                alphas2 = pair.parent2.get_alphas()
                alphas = np.where(alpha_choice, alphas1, alphas2)
            
            offspring_individuals.append(Individual(genotype=genotype, sigmas=sigmas, alphas=alphas))

        return offspring_individuals


class Intermediate(Recombination):
    def __init__(self, offsprings_size: int, selection: Selection):
        self.offsprings_size = offsprings_size
        self.selection = selection

    def recombine(self, population: Population) -> List[Individual]:
        offspring_individuals = []

        pairs = self.selection.select_pairs(population, self.offsprings_size)
        for pair in pairs:
            genotype1 = pair.parent1.get_genotype()
            genotype2 = pair.parent2.get_genotype()
            genotype = np.average(np.array([genotype1, genotype2]), axis=0)
            sigmas = np.array([])
            alphas = np.array([])

            if (len(pair.parent1.get_sigmas()) != 0 and len(pair.parent2.get_sigmas()) != 0):
                sigmas1 = pair.parent1.get_sigmas()
                sigmas2 = pair.parent2.get_sigmas()
                sigmas = np.average(np.array([sigmas1, sigmas2]), axis=0)

            if (len(pair.parent1.get_alphas()) != 0 and len(pair.parent2.get_alphas()) != 0):
                alphas1 = pair.parent1.get_alphas()
                alphas2 = pair.parent2.get_alphas()
                alphas = np.average(np.array([alphas1, alphas2]), axis=0)
            
            offspring_individuals.append(Individual(genotype=genotype, sigmas=sigmas, alphas=alphas))

        return offspring_individuals

class GlobalDiscrete(Recombination):
    def __init__(self, offsprings_size: int, selection: Selection):
        self.offsprings_size = offsprings_size
        self.selection = selection

    def recombine(self, population: Population) -> List[Individual]:
        genotype_size = population.individuals[0].size()

        offsprings = []
        for _ in range(self.offsprings_size):
            genes = []
            sigmas = []
            parents_selected = []
            for index in range(genotype_size):
                pair = self.selection.select_pairs(population, 1)[0]

                parent1_selected = np.random.uniform(0, 1) > 0.5
                
                parent_selected = pair.parent1 if parent1_selected else pair.parent2
                gene = parent_selected.genotype[index]
                parents_selected.append(parent_selected)
                genes.append(gene)
                
                if (len(pair.parent1.get_sigmas()) != 0 and len(pair.parent2.get_sigmas()) != 0):
                    sigma = parent_selected.sigmas[index]
                    sigmas.append(sigma)
   
            alphas = []
            if (len(pair.parent1.get_alphas()) != 0 and len(pair.parent2.get_alphas()) != 0):
                counter = 0
                for i in range(0, genotype_size):
                    for j in range(i+1, genotype_size):
                        alpha1 = parents_selected[i].get_alphas()[counter]
                        alpha2 = parents_selected[j].get_alphas()[counter]
                        alpha_selected = alpha1 if np.random.uniform(0, 1) > 0.5 else alpha2
                        alphas.append(alpha_selected)
                        counter += 1
            
            offsprings.append(Individual(genotype=np.array(genes), sigmas=np.array(sigmas), alphas=np.array(alphas)))

        return offsprings


class GlobalIntermediate(Recombination):
    def __init__(self, offsprings_size: int, selection: Selection):
        self.offsprings_size = offsprings_size
        self.selection = selection

    def recombine(self, population: Population) -> List[Individual]:
        genotype_size = population.individuals[0].size()

        offsprings = []
        for _ in range(self.offsprings_size):
            genes = []
            sigmas = []
            pairs_selected = []
            for index in range(genotype_size):
                pair = self.selection.select_pairs(population, 1)[0]
                pairs_selected.append(pair)

                average_gene = np.average(np.array([pair.parent1.genotype[index], pair.parent2.genotype[index]]), axis=0)
                genes.append(average_gene)
                
                if (len(pair.parent1.get_sigmas()) != 0 and len(pair.parent2.get_sigmas()) != 0):
                    average_sigma = np.average(np.array([pair.parent1.sigmas[index], pair.parent2.sigmas[index]]), axis=0)
                    sigmas.append(average_sigma)
            
            alphas = []
            if (len(pair.parent1.get_alphas()) != 0 and len(pair.parent2.get_alphas()) != 0):
                counter = 0
                for i in range(0, genotype_size):
                    for j in range(i+1, genotype_size):
                        alpha1_1 = pairs_selected[i].parent1.get_alphas()[counter]
                        alpha1_2 = pairs_selected[i].parent2.get_alphas()[counter]
                        alpha2_1 = pairs_selected[j].parent1.get_alphas()[counter]
                        alpha2_2 = pairs_selected[j].parent2.get_alphas()[counter]

                        average_alpha = np.average(np.array([
                            alpha1_1, alpha1_2, alpha2_1, alpha2_2
                            ]), axis=0)

                        alphas.append(average_alpha)
                        counter += 1 
            
            offsprings.append(Individual(genotype=np.array(genes), sigmas=np.array(sigmas), alphas=np.array(alphas)))

        return offsprings


class GlobalWeightedIntermediate(Recombination):
    def __init__(self, offsprings_size: int, selection: Selection):
        self.offsprings_size = offsprings_size
        self.selection = selection

    def recombine(self, population: Population) -> List[Individual]:
        genotype_size = population.individuals[0].size()

        offsprings = []
        for _ in range(self.offsprings_size):
            genes = []
            sigmas = []
            pairs_selected = []
            for index in range(genotype_size):
                pair = self.selection.select_pairs(population, 1)[0]
                pairs_selected.append(pair)

                min_fitness = min(pair.parent1.fitness, pair.parent1.fitness)
                fitness1 = pair.parent1.fitness + min_fitness + 1
                fitness2 = pair.parent2.fitness + min_fitness + 1

                fitnesses_sum = float(fitness1 + fitness2)
                parent1_weight = 1 - (fitness1 / fitnesses_sum)
                parent2_weight = 1 - (fitness2 / fitnesses_sum)
                
                weighted_average_gene = parent1_weight * \
                    pair.parent1.genotype[index] + \
                    parent2_weight * pair.parent2.genotype[index]
                genes.append(weighted_average_gene)
                
                if (len(pair.parent1.get_sigmas()) != 0 and len(pair.parent2.get_sigmas()) != 0):
                    weighted_average_sigma = parent1_weight * \
                        pair.parent1.sigmas[index] + \
                        parent2_weight * pair.parent2.sigmas[index]
                    sigmas.append(weighted_average_sigma)
                
            alphas = []
            if (len(pair.parent1.get_alphas()) != 0 and len(pair.parent2.get_alphas()) != 0):
                counter = 0
                for i in range(0, len(pairs_selected)):
                    for j in range(i+1, len(pairs_selected)):
                        alpha1_1 = pairs_selected[i].parent1.get_alphas()[counter]
                        alpha1_2 = pairs_selected[i].parent2.get_alphas()[counter]
                        alpha2_1 = pairs_selected[j].parent1.get_alphas()[counter]
                        alpha2_2 = pairs_selected[j].parent2.get_alphas()[counter]
                        fitness1_1 = pairs_selected[i].parent1.fitness
                        fitness1_2 = pairs_selected[i].parent2.fitness
                        fitness2_1 = pairs_selected[j].parent1.fitness
                        fitness2_2 = pairs_selected[j].parent2.fitness

                        min_fitness = min(fitness1_1, fitness1_2, fitness2_1, fitness2_2)   
                        fitness1_1 = fitness1_1 + min_fitness + 1
                        fitness1_2 = fitness1_2 + min_fitness + 1
                        fitness2_1 = fitness2_1 + min_fitness + 1
                        fitness2_2 = fitness2_2 + min_fitness + 1

                        fitnesses_sum = float(fitness1_1 + fitness1_2 + fitness2_1 + fitness2_2)
                        parent1_1_weight = 1 - (fitness1_1 / fitnesses_sum)
                        parent1_2_weight = 1 - (fitness1_2 / fitnesses_sum)
                        parent2_1_weight = 1 - (fitness2_1 / fitnesses_sum)
                        parent2_2_weight = 1 - (fitness2_2 / fitnesses_sum)

                        weighted_average_alpha = parent1_1_weight * alpha1_1 +\
                            parent1_2_weight * alpha1_2 +\
                            parent2_1_weight * alpha2_1 +\
                            parent2_2_weight * alpha2_2

                        alphas.append(weighted_average_alpha)
                        counter += 1 
            
            offsprings.append(Individual(genotype=np.array(genes), sigmas=np.array(sigmas), alphas=np.array(alphas)))

        return offsprings
