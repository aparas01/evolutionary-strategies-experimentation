from ioh import get_problem
from ioh import logger
from dataclasses import dataclass
import math
from model import Individual, Population
import numpy as np
from recombination import Discrete, GlobalDiscrete, GlobalIntermediate, GlobalWeightedIntermediate, Intermediate, Recombination
from selection import RandomSelection, Selection
from mutation import CorrelatedSigma, SingleSigma, IndividualSigmaPerGene, Mutation
from enum import Enum
import argparse
from sklearn.metrics import mean_absolute_error
from pathlib import Path
np.random.seed(42)

# Default Parameters
DEFAULT_BUDGET = 50000
POPULATION_SIZE_DEFAULT = 8
OFFSPRINGS_SIZE_DEFAULT = 100
NEXT_GENERATION_SELECTION_DEFAULT = ','
PARENT_SELECTION_DEFAULT = 'random'
RECOMBINATION_DEFAULT = 'global-intermediate'
MUTATION_DEFAULT = 'single-sigma'

# ES algorithm options
RESULTS_DIR = 'es_results'
RECOMBINATIONS = ['discrete', 'intermediate', 'global-discrete',
                  'global-intermediate', 'global-weighted-intermediate']
MUTATIONS = ['single-sigma', 'individual-sigma', 'correlated-sigma']
NEXT_GENERATION_SELECTIONS = ['+', ',']
VERBOSE_CHOICE = [0, 1]


class NextGenerationSelection(Enum):
    OFFSPRINGS = ','
    ALL = '+'

@dataclass
class ESConfiguration():
    population_size: int
    offsprings_size: int
    selection: Selection
    recombination: Recombination
    mutation: Mutation
    next_generation: NextGenerationSelection
    verbose: int

# Mapping from recombination option to implementation class
RECOMBINATION = {
    'discrete': Discrete,
    'intermediate': Intermediate,
    'global-discrete': GlobalDiscrete,
    'global-intermediate': GlobalIntermediate,
    'global-weighted-intermediate': GlobalWeightedIntermediate
}

# Mapping from mutation option to implementation class
MUTATION = {
    'single-sigma': SingleSigma,
    'individual-sigma': IndividualSigmaPerGene,
    'correlated-sigma': CorrelatedSigma
}
NEXT_GENERATION_SELECTIONS = ['+', ',']
VERBOSE_CHOICE = [0, 1]

# Script Arguments
def parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', help='Population size',
                        type=int, default=POPULATION_SIZE_DEFAULT)
    parser.add_argument('--offsprings', help='Offsprings size',
                        type=int, default=OFFSPRINGS_SIZE_DEFAULT)
    parser.add_argument('--recombination', help='Recombination method',
                        choices=RECOMBINATIONS, default=RECOMBINATION_DEFAULT)
    parser.add_argument('--mutation', help='Mutation method',
                        choices=MUTATIONS, default=MUTATION_DEFAULT)
    parser.add_argument('--verbose', help='Show prints',
                        choices=VERBOSE_CHOICE, default=1, type=int)
    parser.add_argument('--nextGeneration', help='Type of next generation selection',
                        choices=NEXT_GENERATION_SELECTIONS, default=NEXT_GENERATION_SELECTION_DEFAULT)

    return parser.parse_args()


next_generation = NextGenerationSelection.ALL

@dataclass
class ESConfiguration():
    population_size: int
    offsprings_size: int
    selection: Selection
    recombination: Recombination
    mutation: Mutation
    next_generation: NextGenerationSelection
    verbose: int


def next_population_from(individuals, size) -> Population:
  sorted_individuals = sorted(
    individuals,
    key=lambda x: x.fitness
  )

  return Population(sorted_individuals[:size])


def ES(func, config:ESConfiguration, budget = DEFAULT_BUDGET):
    individual_size =  func.meta_data.n_variables
    config.mutation.setup(individual_size)
    alpha_len = (int(individual_size*(individual_size-1)/2))
    pop =   Population(
                [Individual(genotype=np.random.uniform(-5,5, size=individual_size), 
                sigmas=np.random.uniform(0.001, 1, individual_size),
                alphas=(np.deg2rad(np.random.uniform(0,360, alpha_len))) if config.mutation.has_alphas() else np.array([]))
                for i in range(config.population_size)]
            )
    budget = pop.evaluate_fitnesses(func, budget)
    while budget > 0 :

        offsprings = config.recombination.recombine(pop)
        
        mutated_offsprings = [config.mutation.mutate(offspring) for offspring in offsprings]
        
        offsprings_pop = Population(mutated_offsprings)
        budget = offsprings_pop.evaluate_fitnesses(func, budget)

        if config.next_generation == NextGenerationSelection.ALL:
            candidate_individuals = pop.individuals.copy() + offsprings_pop.individuals.copy()
        else:
            candidate_individuals = offsprings_pop.individuals.copy()
        
        pop = next_population_from(candidate_individuals, pop.size())
        best_individual = pop.best_individual()

    return best_individual.fitness   


def run_experiment(config:ESConfiguration, run_name:str):
    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress this folder and upload it to IOHanalyzer
    l = logger.Analyzer(root="data", 
        folder_name=run_name, 
        algorithm_name=run_name, 
        algorithm_info="An Evolution Strategy on the 24 BBOB problems in python")

    all = []
    # Testing on 24 problems
    for pid in range(1,25): 
        if(config.verbose):
            print('*************************')
            print('Problem ', pid, ':')
        mse_curr = []

        # Testing 10 instances for each problem
        # for ins in range(1,2):
        for ins in range(1,11):
            if(config.verbose):
                print('\tIteration ', ins, ':')
            
            # Getting the problem with corresponding problem_id, instance_id, and dimension n = 5.
            problem = get_problem(pid, dim = 5, iid = ins, problem_type = 'BBOB')

            # Attach the problem with the logger
            problem.attach_logger(l)

            # The assignment 2 requires only one run for each instance of the problem.
            bf = ES(problem, config)
            bf = round(bf, 3)
            # To reset evaluation information as default before the next independent run.
            # DO NOT forget this function before starting a new run. Otherwise, you may meet issues with uploading data to IOHanalyzer.
            problem.reset()
            optimum = problem.objective.y
            if(config.verbose):
                print('\t\tTarget: ', optimum, '\n\t\tFound:  ', bf, '\n')
            mse_curr.append(mean_absolute_error([optimum], [bf]))
        all.append(np.mean(mse_curr))

    np.save(f'./{RESULTS_DIR}/{run_name}.npy', all)
    # This statemenet is necessary in case data is not flushed yet.

    del l

def run_name_from(args):
    return f'ES_{args.population}{args.nextGeneration}{args.offsprings}_{args.recombination}_{args.mutation}'

if __name__ == '__main__':
    args = parsed_arguments()
    selection = RandomSelection()
    recombination = RECOMBINATION[args.recombination](args.offsprings, selection)
    mutation = MUTATION[args.mutation](1.0)
    next_generation_selection = NextGenerationSelection.ALL if args.nextGeneration == '+' else NextGenerationSelection.OFFSPRINGS
    verbose = args.verbose

    configuration = ESConfiguration(
        population_size=args.population,
        offsprings_size=args.offsprings,
        selection=selection,
        recombination=recombination,
        mutation=mutation,
        next_generation=next_generation_selection,
        verbose=verbose
    )


    Path(f'./{RESULTS_DIR}').mkdir(parents=True, exist_ok=True)
    
    
    try:
        print(f'Execution Started, {run_name_from(args)}')
        run_experiment(configuration, run_name_from(args))
        print(f'Execution Finished, {run_name_from(args)}')
    except Exception as ex:
        print(f'Run {run_name_from(args)} failed with Exception: {ex}')