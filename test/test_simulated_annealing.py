import pytest
import numpy as np
from optimizer.simulated_annealing import SimulatedAnnealing

def objective_function(np_variable):
    return -np.sum(np_variable)

def perturbation_function(np_variable, n_perturbation = 1):
    np_aux_variable = np.copy(np_variable)
    position = np.random.randint(low = 0, high = len(np_aux_variable), size = n_perturbation)
    np_aux_variable[position] = abs(np_aux_variable[position]-1)
    return np_aux_variable

def cooling_function(temperature, factor):
    return temperature * factor

stopping_criteria = {'n_max' : 1e5, 'min_temperature': 1e-9}

n_itens = 100
initial_solution = np.random.binomial(1, .5, n_itens)

initial_temperature = 1e4
cooling_factor = 9.9e-1

def test_simulated_annealing_init():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criteria, initial_solution, cooling_factor, initial_temperature)
    assert "[OK] Simulated annealing created successfully" in optimizer.status

def test_simulated_annealing_wrong_format_objective_function():
    optimizer = SimulatedAnnealing(1, perturbation_function, cooling_function, stopping_criteria, initial_solution, cooling_factor, initial_temperature)
    assert "[ERROR]<objective> A controller function must be a python function object" in optimizer.status

def test_simulated_annealing_missing_objective_function():
    optimizer = SimulatedAnnealing({}, perturbation_function, cooling_function, stopping_criteria, initial_solution, cooling_factor, initial_temperature)
    assert "[ERROR] Invalid objective function" in optimizer.status

def test_simulated_annealing_wrong_format_perturbation_function():
    optimizer = SimulatedAnnealing(objective_function, 1, cooling_function, stopping_criteria, initial_solution, cooling_factor, initial_temperature)
    assert "[ERROR]<perturbation> A controller function must be a python function object" in optimizer.status

def test_simulated_annealing_missing_perturbation_function():
    optimizer = SimulatedAnnealing(objective_function, {}, cooling_function, stopping_criteria, initial_solution, cooling_factor, initial_temperature)
    assert "[ERROR] Invalid perturbation function" in optimizer.status

def test_simulated_annealing_wrong_format_cooling_function():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, 1, stopping_criteria, initial_solution, cooling_factor, initial_temperature)
    assert "[ERROR]<cooling> A controller function must be a python function object" in optimizer.status

def test_simulated_annealing_missing_cooling_function():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, {}, stopping_criteria, initial_solution, cooling_factor, initial_temperature)
    assert "[ERROR] Invalid cooling function" in optimizer.status

def test_simulated_annealing_wrong_format_stopping_criteria_function():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, 1, initial_solution, cooling_factor, initial_temperature)
    assert "[ERROR] The stopping criteria must be a python dict object" in optimizer.status

def test_simulated_annealing_missing_stopping_criteria_function():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, {}, initial_solution, cooling_factor, initial_temperature)
    assert "[ERROR] Invalid stopping criteria" in optimizer.status

def test_simulated_annealing_wrong_format_initial_solution():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criteria, 1, cooling_factor, initial_temperature)
    assert "[ERROR] The initial solution must be a list, pandas series or numpy ndarray" in optimizer.status

def test_simulated_annealing_wrong_format_cooling_factor():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criteria, initial_solution, "1e-6", initial_temperature)
    assert "[ERROR] The cooling factor must be a float or int" in optimizer.status

def test_simulated_annealing_negative_cooling_factor():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criteria, initial_solution, -1, initial_temperature)
    assert "[ERROR] The cooling factor must be positive and less than 1.0" in optimizer.status

def test_simulated_annealing_large_cooling_factor():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criteria, initial_solution, 1, initial_temperature)
    assert "[ERROR] The cooling factor must be positive and less than 1.0" in optimizer.status

def test_simulated_annealing_wrong_initial_temperature():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criteria, initial_solution, cooling_factor,"1e-6")
    assert "[ERROR] The initial temperature must be a float or int" in optimizer.status

def test_simulated_annealing_initial_temperature():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criteria, initial_solution, cooling_factor, -1)
    assert "[ERROR] The initial temperature must be positive" in optimizer.status

def test_simulated_annealing_solution():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criteria, initial_solution, cooling_factor, initial_temperature)
    solution = optimizer.get_solution()
    assert objective_function(solution) == -100.0