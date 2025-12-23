#%%
from deap import base, creator, tools, algorithms
import random
import numpy as np
import wandb



#%%
creator.create("Fitness", base.Fitness, weights=(-1, 1))
creator.create("Individual", set, fitness=creator.Fitness)


#%%
items = {}
NBR_ITEMS = 50
MAX_ITEM = 20
MAX_WEIGHT = 50
IND_INIT_SIZE = 10
#%%
# NBR_ITEMS個のアイテムをランダムに生成
# アイテムは(重さ, 価値)のタプルで表現
# 重さは1から10の整数、価値は0から100の実数
for i in range(NBR_ITEMS):
    items[i] = (random.randint(1, 10), random.uniform(0, 100))

toolbox = base.Toolbox()
toolbox.register("attr_item", random.randrange, NBR_ITEMS)
toolbox.register("Individual", tools.initRepeat, creator.Individual, toolbox.attr_item, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
#%%
def evalKnapsack(individual):
    weight = 0.0
    value = 0.0
    for item in individual:
        weight += items[item][0]
        value += items[item][1]
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        return 100000000, 0
    return weight, value

def cxSet(ind1, ind2):
    temp = set(ind1)
    ind1 &= ind2
    ind2 ^= temp
    return ind1, ind2

def mutSet(individual):
    if random.random() < 0.5:
        if len(individual) > 0:
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))
    return individual,
#%%
toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)
#%%

def inhouse_mu_plus_lambda(population, toolbox, mu, lamnbda_, cxpb, mutpb, ngen, run=None, stats=None, halloffame=None):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if run:
        # 辞書として最新値を送信
        run.log({"gen": 0, "nevals": len(invalid_ind), "weight_avg": record["avg"][0], "weight_min": record["min"][0],
                 "value_avg": record["avg"][1], "value_max": record["max"][1]})
    for gen in range(1, ngen + 1):
        offspring = algorithms.varOr(population, toolbox, lamnbda_, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(population + offspring, mu)

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        print(record)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if run:
            # 世代ごとの差分のみを送信
            run.log({"gen": gen, "nevals": len(invalid_ind), "weight_avg": record["avg"][0], "weight_min": record["min"][0],
                     "value_avg": record["avg"][1], "value_max": record["max"][1]})
    return population, logbook

#%%
def main():
    NGEN = 50
    MU = 50
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2
    run = wandb.init(project="napzack-in-wandb", name="mop-napzack-deap")

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    inhouse_mu_plus_lambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, run=run, stats=stats, halloffame=hof)

    return pop, stats, hof
#%%
if __name__ == "__main__":
    
    random.seed(42)
    main()
# %%