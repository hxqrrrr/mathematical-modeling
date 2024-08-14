import numpy as np
import random


# 定义问题：在这个例子中，我们尝试最大化函数 f(x) = x^2，其中 0 <= x <= 31
def fitness_function(x):
    return x ** 2


# 初始化种群
def initialize_population(pop_size, chromosome_length):
    return np.random.randint(2, size=(pop_size, chromosome_length))


# 解码染色体
def decode_chromosome(chromosome):
    return int(''.join(map(str, chromosome)), 2)


# 选择操作
def selection(population, fitness_values):
    return random.choices(population, weights=fitness_values, k=len(population))


# 交叉操作
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


# 变异操作
def mutation(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome


# 主遗传算法循环
def genetic_algorithm(pop_size, chromosome_length, generations, mutation_rate):
    population = initialize_population(pop_size, chromosome_length)

    for _ in range(generations):
        # 计算适应度
        fitness_values = [fitness_function(decode_chromosome(chrom)) for chrom in population]

        # 选择
        new_population = selection(population, fitness_values)

        # 交叉和变异
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                new_population[i], new_population[i + 1] = crossover(new_population[i], new_population[i + 1])
            new_population[i] = mutation(new_population[i], mutation_rate)
            if i + 1 < pop_size:
                new_population[i + 1] = mutation(new_population[i + 1], mutation_rate)

        population = new_population

    # 返回最佳个体
    best_chromosome = max(population, key=lambda chrom: fitness_function(decode_chromosome(chrom)))
    return decode_chromosome(best_chromosome)


# 运行算法
result = genetic_algorithm(pop_size=50, chromosome_length=5, generations=100, mutation_rate=0.01)
print(f"最优解: x = {result}, f(x) = {fitness_function(result)}")
