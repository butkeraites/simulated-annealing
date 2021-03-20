import numpy as np
import pandas as pd

class SimulatedAnnealing:
    def __init__(self, objective, perturbation, cooling, stopping_criteria, initial_solution, cooling_factor, initial_temperature, heat_again = True):
        self.__initialize_parameters(heat_again)
        self.__initial_solution_aquisition(initial_solution)
        self.__cooling_factor_aquisition(cooling_factor)
        self.__initial_temperature_aquisition(initial_temperature)
        self.__function_aquisition(objective, 'objective')
        self.__function_aquisition(perturbation, 'perturbation')
        self.__function_aquisition(cooling, 'cooling')
        self.__stopping_criteria_aquisition(stopping_criteria)
        self.__final_validation()
        self.__run_optimization()

    def __initialize_parameters(self, heat_again):
        self.status = []
        self.__controller = {
            'stopping_criteria' : []
        }
        self.__parameters = {
            'iteration' : 0,
            'reheat' : heat_again
        }
        self.__command = "GO"

    def __initial_solution_aquisition(self, initial_solution):
        if self.__command == "GO":
            if isinstance(initial_solution,(list,pd.core.series.Series,np.ndarray)):
                self.status.append("[OK] Aquisition of initial solution done successfully")
                self.__solution = np.copy(initial_solution)
            else:
                self.status.append("[ERROR] The initial solution must be a list, pandas series or numpy ndarray")
                self.__command = "NO GO"

    def __cooling_factor_aquisition(self, cooling_factor):
        if self.__command == "GO":
            if isinstance(cooling_factor, (float, int)):
                if cooling_factor < 1.0 and cooling_factor > 0.0:
                    self.status.append("[OK] Aquisition of cooling factor done successfully")
                    self.__cooling_factor = np.float(cooling_factor)
                else:
                    self.status.append("[ERROR] The cooling factor must be positive and less than 1.0")
                    self.__command = "NO GO"
            else:
                self.status.append("[ERROR] The cooling factor must be a float or int")
                self.__command = "NO GO"

    def __initial_temperature_aquisition(self, initial_temperature):
        if self.__command == "GO":
            if isinstance(initial_temperature, (float, int)):
                if initial_temperature > 0.0:
                    self.status.append("[OK] Aquisition of initial temperature done successfully")
                    self.__parameters['temperature'] = np.float(initial_temperature)
                    self.__parameters['initial_temperature'] = np.float(initial_temperature)
                else:
                    self.status.append("[ERROR] The initial temperature must be positive")
                    self.__command = "NO GO"
            else:
                self.status.append("[ERROR] The initial temperature must be a float or int")
                self.__command = "NO GO"


    def __function_aquisition(self, function, function_name):
        if self.__command == "GO":
            if function:
                if hasattr(function, '__call__'):
                    self.status.append("[OK] Aquisition of {} done successfully".format(function_name))
                    self.__controller[function_name] = function
                else:
                    self.status.append("[ERROR]<{}> A controller function must be a python function object".format(function_name))
                    self.__command = "NO GO"
            else:
                self.status.append("[ERROR] Invalid {} function".format(function_name))
                self.__command = "NO GO"

    def __stopping_criteria_aquisition(self, stopping_criteria):
        if self.__command == "GO":
            if stopping_criteria:
                if isinstance(stopping_criteria, dict):
                    if "n_max" in stopping_criteria:
                        self.status.append("[OK] Aquisition of stopping criteria done successfully")
                        self.__parameters['n_max'] = stopping_criteria['n_max']
                        self.__controller['stopping_criteria'].append(self.__halt_by_n_max)
                    if "min_temperature" in stopping_criteria:
                        self.status.append("[OK] Aquisition of stopping criteria done successfully")
                        self.__parameters['min_temperature'] = stopping_criteria['min_temperature']
                        self.__controller['stopping_criteria'].append(self.__halt_by_min_temperature)
                else:
                    self.status.append("[ERROR] The stopping criteria must be a python dict object")
                    self.__command = "NO GO"
            else:
                self.status.append("[ERROR] Invalid stopping criteria")
                self.__command = "NO GO"

    def __final_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Simulated annealing creation failed")
            self.__command = "NO GO"
            self.__stop_motive = "Input data inconsistency"
        else:
            self.status.append("[OK] Simulated annealing created successfully")

    def __halt_by_n_max(self):
        if self.__parameters['n_max'] < self.__parameters['iteration']:
            self.__stop_motive = 'n_max'
        return self.__parameters['n_max'] < self.__parameters['iteration']

    def __reheat(self):
        self.__parameters['temperature'] = np.copy(self.__parameters['initial_temperature'])

    def __halt_by_min_temperature(self):
        if self.__parameters['temperature'] < self.__parameters['min_temperature']:
            if self.__parameters['reheat']:
                self.__reheat()
            else:
                self.__stop_motive = 'min_temperature'
        return self.__parameters['temperature'] < self.__parameters['min_temperature']

    def __new_solution_is_better(self, new_solution):
        return np.less(self.__controller['objective'](new_solution), self.__controller['objective'](self.__solution))

    def __calculate_difference(self, new_solution):
        return np.subtract(self.__controller['objective'](self.__solution), self.__controller['objective'](new_solution))
    
    def __division_of_difference_by_temperature(self, new_solution):
        return np.true_divide(self.__calculate_difference(new_solution),self.__parameters['temperature'])
    
    def __change_solution_even_it_is_bad(self, new_solution):
        alpha = np.random.random_sample()
        boltzman_element = np.exp(self.__division_of_difference_by_temperature(new_solution), dtype = np.float128)
        return np.less_equal(alpha, boltzman_element)

    def __cooling_temperature(self):
        self.__parameters['temperature'] = self.__controller['cooling'](self.__parameters['temperature'], self.__cooling_factor)

    def __run_optimization(self):
        if self.__command == "GO":
            while not any([stopping_criteria() for stopping_criteria in self.__controller['stopping_criteria']]):
                new_solution = self.__controller['perturbation'](self.__solution)
                if (self.__new_solution_is_better(new_solution)) | (self.__change_solution_even_it_is_bad(new_solution)):
                    self.__solution = np.copy(new_solution)
                self.__cooling_temperature()
                self.__parameters['iteration'] += 1

    def get_solution(self):
        return self.__solution
    
    def get_stopping_motive(self):
        return self.__stop_motive