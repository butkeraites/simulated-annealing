import pytest
import numpy as np
from optimizer.simulated_annealing import SimulatedAnnealing

def objective_function(np_variable):
    return np.sum(np_variable)

def test_simulated_annealing_init():
    optimizer = SimulatedAnnealing(objective_function)
    assert "[OK] Simulated annealing created successfully" in optimizer.status

def test_simulated_annealing_wrong_format_objective_function():
    optimizer = SimulatedAnnealing(1)
    assert "[ERROR] Objective function must be a python function object" in optimizer.status

def test_simulated_annealing_missing_objective_function():
    optimizer = SimulatedAnnealing({})
    assert "[ERROR] Invalid objective function" in optimizer.status