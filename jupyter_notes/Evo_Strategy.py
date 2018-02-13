import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1             # DNA (real number)
DNA_BOUND = [0, 5]       # solution upper and lower bounds
N_GENERATIONS = 200
POP_SIZE = 100           # population size
N_KID = 50               # n kids per generation
np.arange()

F = lambda x:np.sin(10*x)*x + np.cos(2*x)*x

def get_fitness(pred): return pred.flatten()

def make_kid(pop,n_kid):
    # generate empty kid holder
    kids = {'DNA':np.empty((n_kid,DNA_SIZE))}