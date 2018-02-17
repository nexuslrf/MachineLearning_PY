
"""
The Evolution Strategy can be summarized as the following term:
{mu/rho +, lambda}-ES
Here we use following term to find a maximum point.
{n_pop/n_pop + n_kid}-ES
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1             # DNA (real number)
DNA_BOUND = [0, 5]       # solution upper and lower bounds
N_GENERATIONS = 200
POP_SIZE = 100           # population size
N_KID = 50               # n kids per generation


F = lambda x:np.sin(10*x)*x + np.cos(2*x)*x

def get_fitness(pred): return pred.flatten()

def make_kid(pop,n_kid):
    # generate empty kid holder
    kids = {'DNA':np.empty((n_kid,DNA_SIZE))}
    kids['mut_strength'] = np.empty_like(kids['DNA'])
    for kv, ks in zip(kids['DNA'],kids['mut_strength']):
        # crossover (roughly half p1, half p2)
        p1, p2 = np.random.choice(np.arange(POP_SIZE),size=2,replace=False)
        cp = np.random.randint(0,2,DNA_SIZE,dtype=np.bool)
        kv[cp] = pop['DNA'][p1,cp]
        kv[~cp] = pop['DNA'][p2,~cp]
        ks[cp] = pop['DNA'][p2,cp]
        ks[~cp] = pop['DNA'][p2,~cp]

        # mutate (change DNA based on normal distribution)
        ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5),0.) # must>0
        kv += ks* np.random.randn(*kv.shape)
        kv[:] = np.clip(kv, *DNA_BOUND) #low & high bound
    return kids

def kill_bad(pop,kids):
    for key in ['DNA','mut_strength']:
        pop[key] = np.vstack((pop[key],kids[key]))

    fitness = get_fitness(F(pop['DNA']))
    idx = np.arange(pop['DNA'].shape[0])
    good_idx = idx[fitness.argsort()][~POP_SIZE:]
    for key in ['DNA','mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop 

pop = dict(DNA=5* np.random.rand(1,DNA_SIZE).repeat(POP_SIZE,axis=0),
    mut_strength=np.random.rand(POP_SIZE,DNA_SIZE))

plt.ion()
x = np.linspace(*DNA_BOUND,200)
plt.plot(x,F(x))

for _ in range(N_GENERATIONS):
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(pop['DNA'],F(pop['DNA']),s=200,lw=0,c='r',alpha=0.5)
    plt.pause(0.1)

    kids = make_kid(pop,N_KID)
    pop = kill_bad(pop, kids)

plt.ioff();
plt.show()