import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    def add_edge(self, u, v):
        self.graph[u][v] = 1
        self.graph[v][u] = 1  # 对于无向图

    def remove_edge(self, u, v):
        self.graph[u][v] = 0
        self.graph[v][u] = 0  # 对于无向图

    def print_graph(self):
        for row in self.graph:
            print(row)


# 创建一个具有5个顶点的图
g = Graph(5)

# 添加一些边
g.add_edge(0, 1)
g.add_edge(0, 4)
g.add_edge(1, 2)
g.add_edge(1, 3)
g.add_edge(1, 4)
g.add_edge(2, 3)
g.add_edge(3, 4)

print("图的邻接矩阵表示：")
g.print_graph()

# 使用NetworkX可视化图
G = nx.Graph()
for i in range(5):
    for j in range(i + 1, 5):
        if g.graph[i][j] == 1:
            G.add_edge(i, j)

nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=16)
plt.title("图的可视化")
plt.show()


# 实现深度优先搜索 (DFS)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=' ')

    for next in range(len(graph.graph[start])):
        if graph.graph[start][next] == 1 and next not in visited:
            dfs(graph, next, visited)


print("\n深度优先搜索结果：")
dfs(g, 0)

# 实现广度优先搜索 (BFS)
from collections import deque


def bfs(graph, start):
    visited = [False] * graph.V
    queue = deque([start])
    visited[start] = True

    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')

        for i in range(len(graph.graph[vertex])):
            if graph.graph[vertex][i] == 1 and not visited[i]:
                queue.append(i)
                visited[i] = True


print("\n\n广度优先搜索结果：")
bfs(g, 0)

# 实现Dijkstra最短路径算法
import sys


def dijkstra(graph, src):
    dist = [sys.maxsize] * graph.V
    dist[src] = 0
    sptSet = [False] * graph.V

    for cout in range(graph.V):
        u = min_distance(dist, sptSet)
        sptSet[u] = True

        for v in range(graph.V):
            if (graph.graph[u][v] > 0 and
                    sptSet[v] == False and
                    dist[v] > dist[u] + graph.graph[u][v]):
                dist[v] = dist[u] + graph.graph[u][v]

    print("\n\nDijkstra最短路径结果（从顶点0开始）：")
    for node in range(graph.V):
        print(f"到顶点 {node} 的最短距离为 {dist[node]}")


def min_distance(dist, sptSet):
    min = sys.maxsize
    for v in range(len(dist)):
        if dist[v] < min and sptSet[v] == False:
            min = dist[v]
            min_index = v
    return min_index


# 为了演示Dijkstra算法，我们需要一个带权重的图
g_weighted = Graph(5)
g_weighted.graph = [
    [0, 2, 0, 6, 0],
    [2, 0, 3, 8, 5],
    [0, 3, 0, 0, 7],
    [6, 8, 0, 0, 9],
    [0, 5, 7, 9, 0]
]

dijkstra(g_weighted, 0)
