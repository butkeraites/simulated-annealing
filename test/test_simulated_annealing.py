import pytest
from optimizer.simulated_annealing import SimulatedAnnealing

def test_simulated_annealing_init():
    optimizer = SimulatedAnnealing()
    assert "[OK] Simulated annealing created successfully" in optimizer.status