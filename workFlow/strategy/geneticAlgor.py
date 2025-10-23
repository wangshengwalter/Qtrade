import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class Single(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        scores = self.mlp(x).squeeze(-1)
        return scores

class GeneticAlgorithm:
    def __init__(self, data_loader, generations=100, pickNum=4, population_size=10, input_dim=16, hidden_dim=64, mutation_rate=0.01):
        self.data_loader = data_loader
        self.generations = generations
        self.pickNum = pickNum
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [Single(input_dim, hidden_dim) for _ in range(population_size)]  # 注意每个个体都是

    def evaluate_fitness(self, individual: nn.Module, data_loader):
        # TODO：：实现适应度评估逻辑
        return 0.0

    def crossover(self, parent1: nn.Module, parent2: nn.Module):
        """
        对两个网络的参数做单点交叉，返回两个新网络实例
        """
        child1 = Single()
        child2 = Single()
        with torch.no_grad():
            for (name, p1), (_, p2), (_, c1), (_, c2) in zip(
                parent1.named_parameters(),
                parent2.named_parameters(),
                child1.named_parameters(),
                child2.named_parameters()
            ):
                flat_p1 = p1.view(-1)
                flat_p2 = p2.view(-1)
                point = torch.randint(1, flat_p1.size(0)-1, (1,)).item()
                new_flat_c1 = torch.cat([flat_p1[:point], flat_p2[point:]])
                new_flat_c2 = torch.cat([flat_p2[:point], flat_p1[point:]])
                c1.data.copy_(new_flat_c1.view_as(c1))
                c2.data.copy_(new_flat_c2.view_as(c2))
        return child1, child2
    
    def mutate(self, individual: nn.Module, mutation_rate=0.01):
        with torch.no_grad():
            for p in individual.parameters():
                mutation_mask = torch.rand_like(p) < mutation_rate
                p[mutation_mask] += torch.randn_like(p)[mutation_mask] * 0.1
        return individual
    
    def run(self):
        best_fitness_over_time = []
        for gen in range(self.generations):
            fitness_scores = [self.evaluate_fitness(ind, self.data_loader) for ind in self.population]
            best_fitness = max(fitness_scores)
            best_fitness_over_time.append(best_fitness)
            print(f"Generation {gen}: Best Fitness = {best_fitness}")

            # 选择适应度最高的个体
            sorted_population = [ind for _, ind in sorted(zip(fitness_scores, self.population), key=lambda x: x[0], reverse=True)]
            selected = sorted_population[:self.pickNum]

            # 生成新一代
            new_population = selected.copy()
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, self.mutation_rate)
                child2 = self.mutate(child2, self.mutation_rate)
                new_population.extend([child1, child2])
            self.population = new_population[:self.population_size]

        # 绘制适应度变化曲线
        plt.plot(best_fitness_over_time)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Fitness Over Generations')
        plt.show()
    






if __name__ == "__main__":

    total_feat = []
    for i in range(2, 17):
        feat = pd.read_pickle(f"score{i}.bin")
        # 给列名加前缀，防止重复
        feat = feat.add_suffix(f'{i-1}')
        total_feat.append(feat)

    # 使用 inner join，对齐索引，防止产生大量NaN（可根据实际情况选择 inner/outer）
    result = pd.concat(total_feat, axis=1, join='inner')
    result.index.set_names(['date', 'stockcode'], inplace=True)
    # 打印column names
    print(result.columns)
    print(result)

    # 检查数据是否有NaN
    if result.isnull().values.any():
        print("数据中存在NaN值，请检查数据完整性。")

    # # 检查数据的每天股票数量是否一致
    # counts = result.groupby(level=0).size()
    # if not counts.nunique() == 1:
    #     print("每天的股票数量不一致，请检查数据。")

    # 存储合并后的数据
    result.to_pickle("stock15feats.bin")
        
    # # 画出每天股票数量的折线图
    # plt.figure(figsize=(12, 6))
    # plt.plot(counts.index, counts.values, marker='o', linewidth=2, markersize=4)
    # plt.title('每日股票数量变化', fontsize=14)
    # plt.xlabel('日期', fontsize=12)
    # plt.ylabel('股票数量', fontsize=12)
    # plt.grid(True, alpha=0.3)
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    
    # # 显示统计信息
    # print(f"股票数量统计:")
    # print(f"最小值: {counts.min()}")
    # print(f"最大值: {counts.max()}")
    # print(f"平均值: {counts.mean():.2f}")
    # print(f"不同数量的天数分布:")
    # print(counts.value_counts().sort_index())
    
    # plt.show()


    
