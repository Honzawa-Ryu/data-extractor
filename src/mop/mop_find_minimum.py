#%%
import deap
import random
from deap import base, creator, tools, algorithms


#%%
# 問題を最小化問題として設定（weightsが負の値）
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# 個体の定義（リストクラスを継承している）
creator.create("Individual", list, fitness=creator.FitnessMin)

# 目的関数の定義
def obfunc(individual):
    x = individual[0]
    y = individual[1]
    object = (x - 1)**2 + (y - 2)**2
    # カンマが必須らしい
    return object,


#%%
# 交叉、突然変異、選択の登録
toolbox = base.Toolbox()

# random.uniformで-50から50の範囲で属性を生成する関数を登録（random.uniformの別名？らしい）
toolbox.register("attribute", random.uniform, -50, 50)
# 個体生成の登録（個体の持つ2つの属性をAttributeで決める）
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=2)
# 集団の個体数を設定するための関数
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# トーナメント形式で次世代に子を残す親を決定
toolbox.register("select", tools.selTournament, tournsize=5)
# 交叉関数の設定、ブレンド交叉という方法を採用
toolbox.register("mate", tools.cxBlend, alpha=0.2)
# 突然変異関数の設定。indbpは各遺伝子が突然変異を起こす確率。muとsigmaは変異の平均と標準偏差
toolbox.register("mutate", tools.mutGaussian, mu=[0.0, 0.0], sigma=[20.0, 20.0], indpb=0.2)
# 目的関数の登録
toolbox.register("evaluate", obfunc)


#%%
random.seed(42)

NGEN = 50 # 世代数
POP = 80 # 集団の個体数
CXPB = 0.9 # 交叉確率
MUTPB = 0.1 # 突然変異確率


#%%
pop = toolbox.population(n=POP)

for individual in pop:
    individual.fitness.values = toolbox.evaluate(individual)

hof = tools.ParetoFront()

algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, halloffame=hof)

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is: ", best_ind)
print("Fitness value: ", best_ind.fitness.values[0])


#%%