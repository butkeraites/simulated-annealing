import numpy as np

class SimulatedAnnealing:
    def __init__(self, objective):
        self.status = []
        self.__objective_aquisition(objective)

    def __objective_aquisition(self, objective):
        if objective:
            if hasattr(objective, '__call__'):
                self.status.append("[OK] Simulated annealing created successfully")
                self.objective = objective
            else:
                self.status.append("[ERROR] Objective function must be a python function object")
        else:
            self.status.append("[ERROR] Invalid objective function")