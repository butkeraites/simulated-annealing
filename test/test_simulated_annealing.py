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

def test_simulated_annealing_init():
    optimizer = SimulatedAnnealing(objective_function, perturbation_function, cooling_function, stopping_criterion)
    assert "[OK] Simulated annealing created successfully" in optimizer.status

def test_simulated_annealing_wrong_format_objective_function():
    optimizer = SimulatedAnnealing(1,perturbation_function, cooling_function, stopping_criterion)
    assert "[ERROR]<objective> A controller function must be a python function object" in optimizer.status

def test_simulated_annealing_missing_objective_function():
    optimizer = SimulatedAnnealing({},perturbation_function, cooling_function, stopping_criterion)
    assert "[ERROR] Invalid objective function" in optimizer.status

def test_simulated_annealing_wrong_format_perturbation_function():
    optimizer = SimulatedAnnealing(objective_function,1, cooling_function, stopping_criterion)
    assert "[ERROR]<perturbation> A controller function must be a python function object" in optimizer.status

def test_simulated_annealing_missing_perturbation_function():
    optimizer = SimulatedAnnealing(objective_function,{}, cooling_function, stopping_criterion)
    assert "[ERROR] Invalid perturbation function" in optimizer.status

def test_simulated_annealing_wrong_format_cooling_function():
    optimizer = SimulatedAnnealing(objective_function,perturbation_function,1,stopping_criterion)
    assert "[ERROR]<cooling> A controller function must be a python function object" in optimizer.status

def test_simulated_annealing_missing_cooling_function():
    optimizer = SimulatedAnnealing(objective_function,perturbation_function,{},stopping_criterion)
    assert "[ERROR] Invalid cooling function" in optimizer.status

def test_simulated_annealing_wrong_format_stopping_criterion_function():
    optimizer = SimulatedAnnealing(objective_function,perturbation_function,cooling_function,1)
    assert "[ERROR]<stopping_criterion> A controller function must be a python function object" in optimizer.status

def test_simulated_annealing_missing_stopping_criterion_function():
    optimizer = SimulatedAnnealing(objective_function,perturbation_function,cooling_function,{})
    assert "[ERROR] Invalid stopping_criterion function" in optimizer.status