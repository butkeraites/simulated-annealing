import numpy as np

class SimulatedAnnealing:
    def __init__(self, objective, perturbation, cooling, stopping_criterion):
        self.status = []
        self.controller = {}
        self.__function_aquisition(objective, 'objective')
        self.__function_aquisition(perturbation, 'perturbation')
        self.__function_aquisition(cooling, 'cooling')
        self.__function_aquisition(stopping_criterion, 'stopping_criterion')
        self.__final_validation()

    def __function_aquisition(self, function, function_name):
        if function:
            if hasattr(function, '__call__'):
                self.status.append("[OK] Aquisition of {} done successfully".format(function_name))
                self.controller[function_name] = function
            else:
                self.status.append("[ERROR]<{}> A controller function must be a python function object".format(function_name))
        else:
            self.status.append("[ERROR] Invalid {} function".format(function_name))

    def __final_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Simulated annealing creation failed")
        else:
            self.status.append("[OK] Simulated annealing created successfully")