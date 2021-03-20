import numpy as np
import pandas as pd
import time
from tqdm import tqdm 

from optimizer.simulated_annealing import SimulatedAnnealing

def large_batch_optimization(n_optimization, n_itens):
    MAX_ITENS = n_itens
    PRODUCT_CATALOG = 2*MAX_ITENS

    def generate_initial_solution():
        initial_solution = np.zeros(PRODUCT_CATALOG)
        return initial_solution
    
    CAPACITY_CATALOG = 1 + np.random.randint(25, size = PRODUCT_CATALOG)
    VALUE_CATALOG = 25 + np.random.randint(25, size = PRODUCT_CATALOG)
    BIG_M = np.max(VALUE_CATALOG) * PRODUCT_CATALOG
    INITIAL_SOLUTION = generate_initial_solution()
    #NUMBER_OF_CHANGES = int(np.ceil(MAX_ITENS*0.3))
    NUMBER_OF_CHANGES = 1

    def backpack_capacity_respected(np_variable):
        return np.dot(np_variable, CAPACITY_CATALOG) <= MAX_ITENS

    def objective_function(np_variable):
        backpack_value = np.dot(np_variable, VALUE_CATALOG)
        if backpack_capacity_respected(np_variable):
            return - backpack_value
        else:
            return - backpack_value + BIG_M
            

    def perturbation_function(np_variable, n_perturbation = NUMBER_OF_CHANGES):
        np_aux_variable = np.copy(np_variable)
        position = np.random.randint(low = 0, high = len(np_aux_variable), size = n_perturbation)
        np_aux_variable[position] = abs(np_aux_variable[position]-1)
        return np_aux_variable

    def cooling_function(temperature, factor):
        return temperature * factor

    def get_solving_status(np_variable):
        if backpack_capacity_respected(np_variable):
            return "feasible"
        else:
            return "infeasible"    

    solutions = {
        'cooling_factor' : [],
        'initial_temperature' : [],
        'final_objective' : [],
        'stop_motive' : [],
        'optimization_status' : [],
        'solution_time' : []
    }

    for i in tqdm(range(int(n_optimization),(int(n_optimization)+1)), desc="Running Simulated Annealing..."):
        start_time = time.time()
        COOLING_FACTOR = np.subtract(1,1e-6)
        INITIAL_TEMPERATURE = np.multiply(1e5,(i+1)**2)
        STOPPING_CRITERIA = {'n_max' : 1e-2*INITIAL_TEMPERATURE, 'min_temperature': 1e-12}

        optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, STOPPING_CRITERIA, INITIAL_SOLUTION, COOLING_FACTOR, INITIAL_TEMPERATURE)
        solution = optimizer.get_solution()

        solutions['cooling_factor'].append(COOLING_FACTOR)
        solutions['initial_temperature'].append(INITIAL_TEMPERATURE)
        solutions['final_objective'].append(objective_function(solution))
        solutions['stop_motive'].append(optimizer.get_stopping_motive())
        solutions['optimization_status'].append(get_solving_status(solution))
        solutions['solution_time'].append(time.time() - start_time)

    results = pd.DataFrame.from_dict(solutions)

    results.to_csv("/home/barbaruiva/Documents/simulated_annealing/results/result_{}_{}.csv".format(str(NUMBER_OF_CHANGES),str(n_itens)))

#for n_optimization in tqdm(range(10,100,10), desc="Iterating over number of trials..."):
n_optimization = 100
n_itens = 50
#for n_itens in tqdm(range(1,100,10), desc="Iterating over itens..."):
large_batch_optimization(n_optimization,n_itens) 