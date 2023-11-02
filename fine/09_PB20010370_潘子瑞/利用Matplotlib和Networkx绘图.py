
import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx
# 创建有向图
G1 = nx.DiGraph()  # 创建一个空的有向图 DiGraph
G1.add_edge('s', '1', capacity=15)  # 添加边的属性 "capacity"，即容量
G1.add_edge('s', '2', capacity=8)
G1.add_edge('1', '2', capacity=20)
G1.add_edge('1', '3', capacity=4)
G1.add_edge('1', 't', capacity=10)
G1.add_edge('2', '3', capacity=15)
G1.add_edge('2', 't', capacity=4)
G1.add_edge('3', 't', capacity=20)

# 求网络最大流
from networkx.algorithms.flow import edmonds_karp  # 导入 edmonds_karp 算法函数
#maxFlowValue：最大流值
#maxFlowDict：达到最大流值的时候各边的流量
maxFlowValue, maxFlowDict = nx.maximum_flow(G1, 's', 't', flow_func=edmonds_karp)  # 用edmonds_karp 算法函数计算网络最大流

# 数据格式转换
edgeCapacity = nx.get_edge_attributes(G1, 'capacity')
edgeLabel = {}  # 边的标签
for i in edgeCapacity.keys():  # 整理边的标签，用于绘图显示
    edgeLabel[i] = f'c={edgeCapacity[i]:}'  # 边的容量
edgeLists = []  # 最大流的边的 list
for i in maxFlowDict.keys():
    for j in maxFlowDict[i].keys():
        edgeLabel[(i, j)] += ',f=' + str(maxFlowDict[i][j])  # 取出每条边流量信息存入边显示值
        if maxFlowDict[i][j] > 0:  # 网络最大流的边（流量>0）
            edgeLists.append((i,j))

# 输出显示
print("最大流值: ", maxFlowValue)
print("最大流的途径及流量: ", maxFlowDict)  # 输出最大流的途径和各路径上的流量
print("最大流的路径：", edgeLists)  # 输出最大流的途径

# 绘制有向网络图
fig, ax = plt.subplots(figsize=(8, 6))
pos = {'s': (1, 8), '1': (1,7), '2': (7, 8), '3': (3, 6), 't': (9, 6) }#指定顶点位置
edge_labels = nx.get_edge_attributes(G1, 'capacity')
ax.set_title("Maximum flow by Networkx and Matplotlib ")  # 设置标题
nx.draw(G1, pos, with_labels=True, node_color='c')  # 绘制有向图，显示顶点标签
nx.draw_networkx_edge_labels(G1, pos, edgeLabel, font_color='navy')  # 绘制网络G的边图，边有label：'capacity' + maxFlow
nx.draw_networkx_edges(G1, pos, edgelist=edgeLists, edge_color='m')  # 绘制网络G的边图，设置指定边的颜色、宽度
plt.axis('on')#Same as true
plt.show()#在窗口中展示这幅图像