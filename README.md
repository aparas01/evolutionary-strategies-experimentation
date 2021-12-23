# Evolutionary strategies

A typical Evolutionary Strategies (ES) framework was implemented in order to experiment with 24 problems belonging to the Black-box Optimization Benchmarking (BBOB) problem suite.

Recombination, mutation were the main components that were implemented in order to experiment with different configurations of our evolutionary strategy. Also, both `μ,λ` and `μ+λ` selection methods are supported.

## Recombination
The following recombination methods were used:
- discrete
- intermediate
- global discrete
- global intermediate
- global weighted intermediate

## Mutation 
The following mutation methods were used:
- single step size mutation
- individual step size (one step size per gene)
- correlated step sizes

## Executing a single experiment
The below command can be used to run a single experiment on the 24 instances of BBOB problems
~~~
$ python3 main.py --population <population_size> --offsprings <offsprings_size> --recombination <recombination_method> --mutation <mutation_method> --nextGeneration <next_generation_method>
~~~

where:
- recombination_method: `discrete`, `intermediate`, `global-discrete`, `global-intermediate`, `global-weighted-intermediate`
- mutation_method: `single-sigma`, `individual-sigma`, `correlated-sigma`
- next_generation_method: `+`, `,`

## Executing experiment in parallel
In order to run experiments with different configurations in parallel, the following script can be modified and executed:
~~~
$ python3 run_experiments.py
~~~

## Handling results
Using the notebook `es_results_handler.ipynb`, we can analyze the mean absolute errors of the runs that were generated in the default `es_results` directory.

Using the IOHAnalyzer the data generated from the `ioh` package can be analyzed further in terms of empirical cumulative distribution function of the running time (ECDF) and the area under the ECDF curve (AUC).

### Collaborators
- [Pavlos Zakkas](https://www.linkedin.com/in/pzakkas/)
- [Andreas Paraskeva](https://www.linkedin.com/in/andreas-paraskeva-2053141a3/)