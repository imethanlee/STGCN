import numpy as np


class EarlyStop:
    def __init__(self, patience: int, minimization: bool = True, tol: float = 1e-5):
        self.cnt = 0
        self.patience = patience
        self.minimization = minimization
        self.tol = -tol
        self.save = False
        if minimization:
            self.best_val = np.inf
        else:
            self.best_val = -np.inf

    def __is_better_than(self, input1, input2):
        if self.minimization:
            return input1 - input2 < self.tol
        else:
            return input2 - input1 < self.tol

    def check(self, val):
        if self.__is_better_than(val, self.best_val):
            self.best_val = val
            self.cnt = 0
            self.save = True
        else:
            self.save = False
        self.cnt += 1
        if self.cnt >= self.patience:
            return True
        return False
