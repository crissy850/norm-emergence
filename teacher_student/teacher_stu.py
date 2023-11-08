import networkx as nx
import numpy as np
import pandas as pd
import coordination_fuction
import matplotlib.pyplot as plt
#较为看重初始化
n = 100
p = 0.1
k = 4
np.random.seed(5)
G = nx.watts_strogatz_graph(n,k,p)#创建小世界网络
decay_rate = 0.01#学习率和策略选择概率的衰减率
actions = ['a', 'b']
action_num = 2
reward_matrix = pd.DataFrame(np.array([1,-1,-1,1]).reshape(action_num,action_num), index=actions, columns=actions)
print(reward_matrix)
a = 0.1
c = 0.9
for node in G.nodes:
    G.nodes[node]['Q_value'] = pd.DataFrame(np.zeros((2,action_num)),columns=actions,index=['row', 'column'])#Q表
    G.nodes[node]['row'] = [0.1,0]#row状态的初始学习率
    G.nodes[node]['column'] = [0.1,0]#column状态的初始学习率
    G.nodes[node]['b'] = 0.1#策略选择概率
    G.nodes[node]['time'] = pd.DataFrame(np.zeros((1,2)),columns=['row','column'])
    G.nodes[node]['b_ask'] = 4000
    G.nodes[node]['b_give'] = 4000
    G.nodes[node]['next_S'] = 'none'
    G.nodes[node]['action'] = 'none'

for k in range(200):
    nums = pd.DataFrame(np.zeros((1, action_num)), columns=actions)
    for i in range(n):
        if k == 0:
            [i_state, j_state] = coordination_fuction.chose_state()
            [r, i_action, j_action, j] = coordination_fuction.interactions(i, actions, reward_matrix, i_state, j_state,
                                                                           decay_rate, G)
            coordination_fuction.upadta_value_average(i, r, i_action, i_state, a, c, G)
        else:
            i_state = G.nodes[i]['next_S']
            if i_state == 'row':
                j_state = 'column'
            else:
                j_state = 'row'
            i_action = coordination_fuction.teacheraction_choose(i,i_state,actions,action_num,G)
            if i_action == 'none':
                [r, i_action, j_action, j] = coordination_fuction.interactions(i, actions, reward_matrix, i_state,
                                                                               j_state, decay_rate, G)
                coordination_fuction.upadta_value_average(i, r, i_action, i_state, a, c, G)
            else:
                j = coordination_fuction.chose_N(i,G)
                j_action = coordination_fuction.choose_one_action(j,j_state,actions,G)
                if i_state == 'row':
                    r = reward_matrix.loc[i_action, j_action]
                else:
                    r = reward_matrix.loc[j_action, i_action]
                coordination_fuction.upadta_value_average(i,r,i_action,i_state,a,c,G)
        nums.loc[0, i_action] += 1
        G.nodes[i]['action'] = i_action
    print(max(nums.loc[0].values))
    if max(nums.loc[0].values) >= 90:
        print(k + 1)
        break
coordination_fuction.show_Q(n,G)
lables = {i:G.nodes[i]['action'] for i in G.nodes}
pos = nx.spring_layout(G)
nx.draw(G,pos,with_labels=False)
nx.draw_networkx_labels(G,pos,labels=lables,verticalalignment='center')
plt.show()