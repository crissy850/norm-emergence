import numpy as np
import pandas as pd
AGENT_NUMBER = 100#agents数量
ACTIONS = ['left', 'right']#行动集合
ACTION_NUMBER = 2#数量
K = 2#博弈人数
M = 5#HCR积累轮数
Reword_list = [1,-1,-1,1]
Reword_MATRIX = pd.DataFrame(np.array(Reword_list).reshape((ACTION_NUMBER,ACTION_NUMBER)),index=ACTIONS,columns=ACTIONS)#奖励矩阵
CRW = pd.DataFrame(np.zeros((AGENT_NUMBER,ACTION_NUMBER)),columns=ACTIONS)#积累奖励
MAX_EPISODES = 2000#最大轮数
AGENT_ACTIONS = pd.DataFrame(np.zeros((AGENT_NUMBER,M)))#最近5次的行动
np.random.seed(5)
def Choose_Agents(agent_number):
    chosen_agents = np.random.randint(low=0,high=AGENT_NUMBER,size=agent_number)
    print(chosen_agents)
    return chosen_agents
def HCR():
    agents_chosen_number = pd.Series(np.zeros((AGENT_NUMBER)))
    for i in range(MAX_EPISODES):
        chosen_agents = Choose_Agents(K)
        chosen_agents_actions = []
        for j in range(K):
            if agents_chosen_number.loc[chosen_agents[j]] <= M:
                agents_chosen_number.loc[chosen_agents[j]]=agents_chosen_number.loc[chosen_agents[j]]+1
                chosen_agents_actions.append(np.random.choice(ACTIONS))
            else:
                agents_chosen_number.loc[chosen_agents[j]] = agents_chosen_number.loc[chosen_agents[j]]+1
                chosen_agents_actions.append(pd.Series.idxmax(CRW.loc[chosen_agents[j]]))
        reword = Reword_MATRIX.loc[chosen_agents_actions[0],chosen_agents_actions[1]]
        for j in range(K):
            CRW.loc[chosen_agents[j],chosen_agents_actions[j]] = CRW.loc[chosen_agents[j],chosen_agents_actions[j]]+reword
            AGENT_ACTIONS.loc[chosen_agents[j],(agents_chosen_number[chosen_agents[j]]-1)%M] = chosen_agents_actions[j]
HCR()
print(AGENT_ACTIONS)
print(CRW)
