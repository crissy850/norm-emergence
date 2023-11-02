import networkx as nx
import numpy as np
import pandas as pd

np.random.seed(5)
# 使用Watts-Strogatz模型生成小世界网络
n = 100  # 节点数量
k = 4  # 每个节点的邻居数目
p = 0.1  # 重连概率
G = nx.watts_strogatz_graph(n, k, p)
fast_a = 0.04
low_a = 0.01
# 初始化所有节点，设置属性left和right为0
for i in range(n):
    G.nodes[i]['left'] = 0
    G.nodes[i]['right'] = 0
    G.nodes[i]['lv'] = 1/2
    G.nodes[i]['rv'] = 1/2
# 打印出所有节点及其属性值
for node in G.nodes(data=True):
    print(node)


def q_learning(n):
    actions = G.nodes[n].copy()
    del actions['lv']
    del actions['rv']
    action = max(actions.items(), key=lambda x: x[1])[0]
    return action

def wolf_phc(n,q,action):
    error = G.nodes[n]['lv']*G.nodes[n]['left'] + G.nodes[n]['rv']*G.nodes[n]['right'] - q
    if error > 0 and action == 'left':
        G.nodes[n]['lv'] += fast_a
    elif error > 0 and action == 'right':
        G.nodes[n]['rv'] += fast_a
    elif error < 0 and action == 'left':
        G.nodes[n]['lv'] += low_a
    else:
        G.nodes[n]['rv'] += low_a
    temp_total = G.nodes[n]['lv'] + G.nodes[n]['rv']
    G.nodes[n]['lv'] = G.nodes[n]['lv'] / temp_total
    G.nodes[n]['rv'] = G.nodes[n]['rv'] / temp_total

reword = pd.DataFrame(np.array([1, -1, -1, 1]).reshape((2, 2)), index=['left', 'right'], columns=['left', 'right'])
for t in range(1000):
    total_l_number = 0
    total_r_number = 0
    for i in range(100):
        l_number = 0
        r_number = 0
        actions = []
        for j in list(G.neighbors(i)):
            p = np.random.rand()
            if p <= 0.1 or G.nodes[j]['left'] == G.nodes[j]['right']:
                action = np.random.choice(['left', 'right'], replace=True)
            else:
                action = np.random.choice(['left','right'],p=[G.nodes[j]['lv'],G.nodes[j]['rv']],replace=True)
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
        r = 0
        for j in actions:
            r += reword.loc[fnal_action, j]
        G.nodes[i][fnal_action] += 0.1 * (r + 0.9 * G.nodes[i][q_learning(i)] - G.nodes[i][fnal_action])
        k = 0
        for j in list(G.neighbors(i)):
            G.nodes[j][actions[k]] += 0.1 * (r + 0.9 * G.nodes[j][q_learning(j)] - G.nodes[j][actions[k]])
            wolf_phc(j,G.nodes[j][actions[k]],actions[k])
            k += 1
        wolf_phc(i,G.nodes[i][fnal_action],fnal_action)
    print(total_l_number,total_r_number)

    if total_l_number >= 90:
        print('left', t + 1)
        break
    if total_r_number >= 90:
        print('right', t + 1)
        break
for node in G.nodes(data=True):
    print(node)