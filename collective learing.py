import networkx as nx
import numpy as np
import numpy.random
import pandas as pd
import math
n = 100
p = 0.1
k = 4
np.random.seed(5)
G = nx.watts_strogatz_graph(n=100,k=4,p=0.1)#创建小世界网络
decay_rate = 0.05#学习率和策略选择概率的衰减率
reword_matrix = pd.DataFrame(np.array([11,-30,0,-30,7,6,0,0,5]).reshape((3,3)),index=['a','b','c']
                             ,columns=['a','b','c'])
print(reword_matrix)
for node in G.nodes:
    G.nodes[node]['Q_value'] = pd.DataFrame(np.zeros((2,3)),columns=['a','b','c'],index=['row', 'column'])#Q表
    G.nodes[node]['row'] = [0.8,0]#row状态的初始学习率
    G.nodes[node]['column'] = [0.8,0]#column状态的初始学习率
    G.nodes[node]['b'] = 0.1#策略选择概率
    G.nodes[node]['s'] = ['none', 0,'none']#agent的action，reward，state
#随机选择智能体交互
def chose_N(i):
    j = np.random.choice(list(G.neighbors(i)),size=1,replace=True)[0]
    return j
#随机选择状态
def chose_state():
    states = np.random.choice(['row', 'column'],size=2,replace=False)
    return states
#更新学习率和策略选择概率
def ab_upadte(i,state):
    G.nodes[i][state][1] += 1
    G.nodes[i][state][0] = G.nodes[i][state][0]*math.exp(-decay_rate*G.nodes[i][state][1])
    G.nodes[i]['b'] = G.nodes[i]['b']*math.exp(-decay_rate*(G.nodes[i]['row'][1]+
                                       G.nodes[i]['column'][1]))
#选择动作
def choose_actions(i, j,i_state,j_state):
    p = np.random.rand(2)
    if p[0] < 1 - G.nodes[i]['b']:
        row_values = G.nodes[i]['Q_value'].loc[i_state].values
        max_value = max(row_values)
        actions = list(G.nodes[i]['Q_value'].columns[row_values == max_value])
        if len(actions) == 1:
            i_action = actions[0]
        else:
            i_action = np.random.choice(actions,size=1)[0]
    else:
        i_action = np.random.choice(['a', 'b', 'c'],size=1,replace=True)[0]
    if p[1] < 1-G.nodes[j]['b']:
        row_values = G.nodes[j]['Q_value'].loc[j_state].values
        max_value = max(row_values)
        actions = list(G.nodes[j]['Q_value'].columns[row_values == max_value])
        if len(actions) == 1:
            j_action = actions[0]
        else:
            j_action = np.random.choice(actions,size=1)[0]
    else:
        j_action = np.random.choice(['a', 'b', 'c'],size=1,replace=True)[0]
    return [i_action,j_action]
#进行交互
def interactions():
    for i in range(100):
        j = chose_N(i)
        [i_state, j_state] = chose_state()
        [i_action,j_action] = choose_actions(i,j,i_state,j_state)
        if i_state == 'row':
            r = reword_matrix.loc[i_action, j_action]
        else:
            r = reword_matrix.loc[j_action, i_action]
        G.nodes[i]['s'][0] = i_action
        G.nodes[i]['s'][1] = r
        G.nodes[i]['s'][2] = i_state
        ab_upadte(i,i_state)
#更新Q值
def upadta_value():
    for i in range(100):
        S = pd.DataFrame(columns=['action','reword','state'])
        k = 0
        for j in list(G.neighbors(i)):
            if G.nodes[j]['s'][2] == G.nodes[i]['s'][2]:
                S.loc[k] = G.nodes[j]['s']
                k += 1
        S.loc[k] = G.nodes[i]['s']
        max_r = S['reword'].max()
        index = S['reword'].idxmax()
        action = S.loc[index, 'action']
        state = S.loc[index, 'state']
        count = S['reword'].value_counts()[max_r]
        G.nodes[i]['Q_value'].loc[state,action] += G.nodes[i][state][0]*(max_r*count/(G.degree(i)+1)
                                                                         -G.nodes[i]['Q_value'].loc[state,action])
#展示所有智能体的Q表
def show_Q():
    for i in range(100):
        print('number:{}'.format(i))
        print(G.nodes[i]['Q_value'])
#开始训练
for i in range(1000):
    num = pd.DataFrame(np.zeros((1,3)),columns=['a', 'b', 'c'])
    interactions()
    upadta_value()
    for j in range(100):
        num.loc[0, G.nodes[j]['s'][0]] += 1
    print(num.loc[0, 'a'])
    if num.loc[0, 'a'] >= 90:
        print(i+1)
        break
show_Q()

