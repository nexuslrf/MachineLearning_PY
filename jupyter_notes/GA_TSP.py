import matplotlib.pyplot as plt
import numpy as np

N_CITES = 20 # DNA size
CROSS_RATE = 0.4
MUTATE_RATE = 0.1
POP_SIZE = 500
N_GENERATIONS = 200

class TravelSalesPerson(object):
    def __init__(self,n_cites):
        self.city_position = np.random.rand(n_cites,2)
        plt.ion()

    def plotting(self,lx,ly,total_d):
        plt.cla()
        plt.scatter(self.city_position[:,0].T,self.city_position[:,1].T,s=100,c='k')
        plt.plot(lx.T,ly.T,'r-')
        plt.text(-0.05,-0.05,'total distance=%.3f'%total_d,fontdict={'size':20,'color':'red'})
        plt.xlim((-0.1,1.1))
        plt.ylim((-0.1,1.1))
        plt.pause(0.01)
    
class GA(object):
    def __init__(self,DNA_size,pop_size,cross_rate,mutate_rate):
        self.DNA_size = DNA_size
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutate_rate =mutate_rate

        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

    def translateDNA(self,DNA,city_position):
        line_x = np.empty_like(DNA,dtype=np.float64)
        line_y = np.empty_like(DNA,dtype=np.float64)
        for i,d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i,:] = city_coord[:,0]
            line_y[i,:] = city_coord[:,1]
        return line_x,line_y
    def get_fitness(self,line_x,line_y):
        total_distance = np.empty((line_x.shape[0],),dtype=np.float64)
        for i,(xs,ys) in enumerate(zip(line_x,line_y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)+np.square(np.diff(ys)))))
        fitness = np.exp(self.DNA_size*2/total_distance) #try to make differences obvious
        return fitness, total_distance

    def select(self,fitness):
        idx = np.random.choice(np.arange(self.pop_size),size=self.pop_size,replace=True,p=fitness/np.sum(fitness))
        return self.pop[idx]

# tricky process
    def crossover(self,parent,pop):
        if np.random.rand()< self.cross_rate:
            i_ = np.random.randint(0,self.pop_size,size=1)
            cross_points = np.random.randint(0,2,size=self.DNA_size).astype(np.bool)
            keep_cites = parent[~cross_points]
            swap_cites = pop[i_,np.isin(pop[i_].ravel(),keep_cites,invert=True)]
            parent[:] = np.concatenate((keep_cites,swap_cites))
        return parent

    def mutate(self,child):
        for point in range(self.DNA_size):
            if np.random.rand()<self.mutate_rate:
                swap_point = np.random.randint(0,self.DNA_size)
                child[swap_point], child[point] = child[point],child[swap_point]
        return child

    def evolve(self,fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent,pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop

ga = GA(N_CITES, POP_SIZE, CROSS_RATE, MUTATE_RATE)

env = TravelSalesPerson(N_CITES)
for generation in range(N_GENERATIONS):
    lx,ly = ga.translateDNA(ga.pop,env.city_position)
    fitness, total_distance = ga.get_fitness(lx,ly)
    ga.evolve(fitness)
    best_idx = np.argmax(fitness)
    print('GEN:',generation,'| best fit:%.2f'%fitness[best_idx])

    env.plotting(lx[best_idx],ly[best_idx],total_distance[best_idx])

plt.ioff()
plt.show()

