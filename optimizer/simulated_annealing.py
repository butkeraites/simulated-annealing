import numpy as np
import pandas as pd

class SimulatedAnnealing:
    def __init__(self, objective, perturbation, cooling, stopping_criterion, initial_solution, cooling_factor):
        self.status = []
        self.controller = {}
        self.command = "GO"
        self.__initial_solution_aquisition(initial_solution)
        self.__cooling_factor_aquisition(cooling_factor)
        self.__function_aquisition(objective, 'objective')
        self.__function_aquisition(perturbation, 'perturbation')
        self.__function_aquisition(cooling, 'cooling')
        self.__function_aquisition(stopping_criterion, 'stopping_criterion')
        self.__final_validation()

    def __initial_solution_aquisition(self, initial_solution):
        if self.command == "GO":
            if isinstance(initial_solution,(list,pd.core.series.Series,np.ndarray)):
                self.status.append("[OK] Aquisition of initial solution done successfully")
                self.initial_solution = np.ndarray(initial_solution)
            else:
                self.status.append("[ERROR]<initial solution> The initial solution must be a list, pandas series or numpy ndarray")
                self.command = "NO GO"

    def __cooling_factor_aquisition(self, cooling_factor):
        if self.command == "GO":
            if isinstance(cooling_factor, (float, int)):
                if cooling_factor < 1.0 and cooling_factor > 0.0:
                    self.status.append("[OK] Aquisition of cooling factor done successfully")
                    self.cooling_factor = np.float(cooling_factor)
                else:
                    self.status.append("[ERROR] The cooling factor must be positive and less than 1.0")
                    self.command = "NO GO"
            else:
                self.status.append("[ERROR]<cooling factor> The cooling factor must be a float or int")
                self.command = "NO GO"

    def __function_aquisition(self, function, function_name):
        if self.command == "GO":
            if function:
                if hasattr(function, '__call__'):
                    self.status.append("[OK] Aquisition of {} done successfully".format(function_name))
                    self.controller[function_name] = function
                else:
                    self.status.append("[ERROR]<{}> A controller function must be a python function object".format(function_name))
                    self.command = "NO GO"
            else:
                self.status.append("[ERROR] Invalid {} function".format(function_name))
                self.command = "NO GO"

    def __final_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Simulated annealing creation failed")
            self.command = "NO GO"
        else:
            self.status.append("[OK] Simulated annealing created successfully")