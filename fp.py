import networkx as nx
import numpy as np
import pandas as pd
np.random.seed(5)
# 使用Watts-Strogatz模型生成小世界网络
n = 100  # 节点数量
k = 4  # 每个节点的邻居数目
p = 0.1  # 重连概率
G = nx.watts_strogatz_graph(n, k, p)

# 初始化所有节点，设置属性left和right为0
for i in range(n):
    G.nodes[i]['left'] = 0
    G.nodes[i]['right'] = 0

# 打印出所有节点及其属性值
for node in G.nodes(data=True):
    print(node)


reword = pd.DataFrame(np.array([1,-1,-1,1]).reshape((2,2)),index=['left','right'],columns=['left','right'])

def fp(n,m):
    m_norm = G.nodes[m]['left'] + G.nodes[m]['right']
    m_norm_left = G.nodes[m]['left']/m_norm
    m_norm_right = G.nodes[m]['right']/m_norm
    norm = [m_norm_left,m_norm_right]
    action = np.argmax(reword @ norm)
    if action == 0:
        action = 'left'
    else:
        action = 'right'
    G.nodes[n][action] += 1
    return action


for t in range(1000):
    total_l_number = 0
    total_r_number = 0
    for i in range(100):
        l_number = 0
        r_number = 0
        actions = []
        for j in list(G.neighbors(i)):
            p = np.random.rand()
            if p <= 0.1 or G.nodes[i]['left'] + G.nodes[i]['right'] == 0:
                action = np.random.choice(['left','right'],replace=True)
                G.nodes[j][action] += 1
            else:
                action = fp(j,i)
            actions.append(action)
            if action == 'left':
                l_number = l_number + 1
            else:
                r_number = r_number + 1
        if l_number >= r_number:
            fnal_action = 'left'
            total_l_number = total_l_number + 1
        else:
            fnal_action = 'right'
            total_r_number = total_r_number + 1
        G.nodes[i][fnal_action] += 1
    for node in G.nodes(data=True):
        print(node)
    if total_l_number >= 90:
        print('left',t+1)
        break
    if total_r_number >=90:
        print('right',t+1)
        break
