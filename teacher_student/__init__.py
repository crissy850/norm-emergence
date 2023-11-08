import numpy as np
import math
import pandas as pd
#随机选择智能体交互
def chose_N(i,G):
    j = np.random.choice(list(G.neighbors(i)),size=1,replace=True)[0]
    return j
#随机选择状态
def chose_state():
    states = np.random.choice(['row', 'column'],size=2,replace=False)
    return states
#更新学习率和策略选择概率
def ab_upadte(i,state,decay_rate,G):
    G.nodes[i][state][1] += 1
    G.nodes[i][state][0] = G.nodes[i][state][0]*math.exp(-decay_rate*G.nodes[i][state][1])
    G.nodes[i]['b'] = G.nodes[i]['b']*math.exp(-decay_rate*(G.nodes[i]['row'][1]+
                                       G.nodes[i]['column'][1]))
#选择动作
def choose_actions(i, j,i_state,j_state,all_actions, G):
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
        i_action = np.random.choice(all_actions,size=1,replace=True)[0]
    if p[1] < 1-G.nodes[j]['b']:
        row_values = G.nodes[j]['Q_value'].loc[j_state].values
        max_value = max(row_values)
        actions = list(G.nodes[j]['Q_value'].columns[row_values == max_value])
        if len(actions) == 1:
            j_action = actions[0]
        else:
            j_action = np.random.choice(actions,size=1)[0]
    else:
        j_action = np.random.choice(all_actions,size=1,replace=True)[0]
    return [i_action,j_action]
#进行交互
def interactions(i,all_actions,reward_matrix,i_state,j_state,decay_rate,G):
    j = chose_N(i,G)
    [i_action,j_action] = choose_actions(i,j,i_state,j_state,all_actions,G)
    if i_state == 'row':
        r = reward_matrix.loc[i_action, j_action]
    else:
        r = reward_matrix.loc[j_action, i_action]
    next_state = np.random.choice(['row','column'],size=1)[0]
    G.nodes[i]['next_S'] = next_state
    G.nodes[i]['time'].loc[0,i_state] += 1
    ab_upadte(i,i_state,decay_rate,G)
    return [r,i_action,j_action,j]
#更新Q值,未用到a，b逐渐降低
def upadta_value_average(i, r, i_action, i_state, a, c, G):
    v = 0
    total_degree = G.degree(i)
    for j in list(G.neighbors(i)):
        total_degree += G.degree(j)
    next_state = G.nodes[i]['next_S']
    row_value = G.nodes[i]['Q_value'].loc[next_state].values
    max_q = max(row_value)
    actions = G.nodes[i]['Q_value'].columns[row_value == max_q]
    actions = list(actions)
    if len(actions) == 1:
        max_actions = actions[0]
    else:
        max_actions = np.random.choice(actions,size=1)[0]
    v += G.degree(i)/total_degree*max_q
    for j in list(G.neighbors(i)):
        v += G.degree(j)/total_degree*G.nodes[j]['Q_value'].loc[next_state,max_actions]
    G.nodes[i]['Q_value'].loc[i_state,i_action] += a*(r + c*v-G.nodes[i]['Q_value'].loc[i_state,i_action])
#展示q表
def show_Q(n, G):
    for i in range(n):
        print('number:{}'.format(i))
        print(G.nodes[i]['Q_value'])
#师生机制
def teacheraction_choose(i,i_state,actions,actions_num,G):
    G.nodes[i]['time'].loc[0,i_state] += 1
    max_i_q = max(G.nodes[i]['Q_value'].loc[i_state].values)
    c_i = G.nodes[i]['time'].loc[0,i_state]
    c_i = math.sqrt(c_i)
    p_ask = math.pow(1+max_i_q,-c_i)
    b_ask = G.nodes[i]['b_ask']
    if b_ask > 0:
        p = np.random.rand()
        if p <= p_ask:
            give_actions = pd.DataFrame(np.zeros((1,actions_num)),columns=actions)
            for j in list(G.neighbors(i)):
                max_j_q = max(G.nodes[j]['Q_value'].loc[i_state].values)
                c_j = G.nodes[j]['time'].loc[0,i_state]
                b_give = G.nodes[j]['b_give']
                if c_j != 0:
                    c_j = math.log(c_j,2)
                    p_give = 1-math.pow(1+max_j_q,-c_j)
                    p = np.random.rand()
                    if p <= p_give and b_give > 0:
                        row_values = G.nodes[j]['Q_value'].loc[i_state].values
                        actions = G.nodes[j]['Q_value'].columns[row_values == max_j_q]
                        actions = list(actions)
                        if len(actions) == 1:
                            action = actions[0]
                        else:
                            action = np.random.choice(actions,size=1)[0]
                        give_actions.loc[0,action] += G.degree(j)
                        G.nodes[j]['b_give'] -= 1
            if give_actions.loc[0].sum() > 0:
                b_ask -= 1
                row_values = give_actions.loc[0].values
                max_value = max(row_values)
                actions = give_actions.columns[row_values == max_value]
                actions = list(actions)
                final_action = np.random.choice(actions,size=1)[0]
                return final_action
    return 'none'
#选择一个动作
def choose_one_action(i,i_state,all_actions,G):
    p = np.random.rand()
    if p < 1 - G.nodes[i]['b']:
        row_values = G.nodes[i]['Q_value'].loc[i_state].values
        max_value = max(row_values)
        actions = list(G.nodes[i]['Q_value'].columns[row_values == max_value])
        if len(actions) == 1:
            i_action = actions[0]
        else:
            i_action = np.random.choice(actions, size=1)[0]
    else:
        i_action = np.random.choice(all_actions, size=1, replace=True)[0]
    return i_action


