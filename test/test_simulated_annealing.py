import pytest
import numpy as np
from optimizer.simulated_annealing import SimulatedAnnealing

def objective_function(np_variable):
    return np.sum(np_variable)

def perturbation_function(np_variable, n_perturbation):
    position = np.random.randint(len(np_variable), size=n_perturbation)
    np_variable[position] = abs(np_variable[position]-1)
    return np_variable

def cooling_function(temperature, factor):
    return temperature * factor

def stopping_criterion(n_iteration, n_max):
    return n_iteration > n_max

n_itens = 10
initial_solution = np.random.binomial(1, .5, n_itens)

cooling_factor = 1e-6

def test_simulated_annealing_init():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criterion, initial_solution, cooling_factor)
    assert "[OK] Simulated annealing created successfully" in optimizer.status

def test_simulated_annealing_wrong_format_objective_function():
    optimizer = SimulatedAnnealing(1, perturbation_function, cooling_function, stopping_criterion, initial_solution, cooling_factor)
    assert "[ERROR]<objective> A controller function must be a python function object" in optimizer.status

def test_simulated_annealing_missing_objective_function():
    optimizer = SimulatedAnnealing({}, perturbation_function, cooling_function, stopping_criterion, initial_solution, cooling_factor)
    assert "[ERROR] Invalid objective function" in optimizer.status

def test_simulated_annealing_wrong_format_perturbation_function():
    optimizer = SimulatedAnnealing(objective_function, 1, cooling_function, stopping_criterion, initial_solution, cooling_factor)
    assert "[ERROR]<perturbation> A controller function must be a python function object" in optimizer.status

def test_simulated_annealing_missing_perturbation_function():
    optimizer = SimulatedAnnealing(objective_function, {}, cooling_function, stopping_criterion, initial_solution, cooling_factor)
    assert "[ERROR] Invalid perturbation function" in optimizer.status

def test_simulated_annealing_wrong_format_cooling_function():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, 1, stopping_criterion, initial_solution, cooling_factor)
    assert "[ERROR]<cooling> A controller function must be a python function object" in optimizer.status

def test_simulated_annealing_missing_cooling_function():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, {}, stopping_criterion, initial_solution, cooling_factor)
    assert "[ERROR] Invalid cooling function" in optimizer.status

def test_simulated_annealing_wrong_format_stopping_criterion_function():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, 1, initial_solution, cooling_factor)
    assert "[ERROR]<stopping_criterion> A controller function must be a python function object" in optimizer.status

def test_simulated_annealing_missing_stopping_criterion_function():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, {}, initial_solution, cooling_factor)
    assert "[ERROR] Invalid stopping_criterion function" in optimizer.status

def test_simulated_annealing_wrong_format_initial_solution():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criterion, 1, cooling_factor)
    assert "[ERROR]<initial solution> The initial solution must be a list, pandas series or numpy ndarray" in optimizer.status

def test_simulated_annealing_wrong_format_cooling_factor():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criterion, initial_solution, "1e-6")
    assert "[ERROR]<cooling factor> The cooling factor must be a float or int" in optimizer.status

def test_simulated_annealing_negative_cooling_factor():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criterion, initial_solution, -1)
    assert "[ERROR] The cooling factor must be positive and less than 1.0" in optimizer.status

def test_simulated_annealing_large_cooling_factor():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criterion, initial_solution, 1)
    assert "[ERROR] The cooling factor must be positive and less than 1.0" in optimizer.status