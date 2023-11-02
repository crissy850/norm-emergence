import numpy as np
import  pandas as pd
ACTIONS = ['cooperation','betrayal']
ACTIONS_NUMBER = 2
REWARD_LIST = [1,-1,-1,1]
REWARD_MATRIX = pd.DataFrame(np.array(REWARD_LIST).reshape(ACTIONS_NUMBER,ACTIONS_NUMBER),index=ACTIONS,columns=ACTIONS)
AGENT_NUMBER = 15
CRW = pd.DataFrame(np.zeros((AGENT_NUMBER,ACTIONS_NUMBER)),columns=ACTIONS)
k = 2
M = 30
MAX_EPISODES =3100
np.random.seed(5)
def Choose_Agents():
    chosen_agents = np.random.randint(low=0,high=AGENT_NUMBER,size=k)
    return chosen_agents
def HCR():
    agents_chosen_actions = pd.DataFrame(np.zeros((AGENT_NUMBER,M)))
    agents_chosen_number = pd.Series(np.zeros(AGENT_NUMBER))
    agents_chosen_reward = pd.DataFrame(np.zeros((AGENT_NUMBER,M)))
    for i in range(MAX_EPISODES):
        chosen_agents = Choose_Agents()
        chosen_actions = []
        for j in range(k):
            if agents_chosen_number.loc[chosen_agents[j]] <= M or CRW.loc[chosen_agents[j]].all() == 0:
                chosen_actions.append(np.random.choice(ACTIONS))
                agents_chosen_number.loc[chosen_agents[j]] = agents_chosen_number.loc[chosen_agents[j]] + 1
            else:
                agents_chosen_number.loc[chosen_agents[j]] = agents_chosen_number.loc[chosen_agents[j]] + 1
                chosen_actions.append(pd.Series.idxmax(CRW.loc[chosen_agents[j]]))
        agents_chosen_reward.loc[chosen_agents[0],(agents_chosen_number[chosen_agents[0]]-1)%M]=\
        REWARD_MATRIX.loc[chosen_actions[0],chosen_actions[1]]
        agents_chosen_reward.loc[chosen_agents[1],(agents_chosen_number[chosen_agents[1]]-1)%M]=\
        REWARD_MATRIX.loc[chosen_actions[1],chosen_actions[0]]
        CRW.loc[chosen_agents[0], chosen_actions[0]] += REWARD_MATRIX.loc[chosen_actions[0],chosen_actions[1]]
        CRW.loc[chosen_agents[1], chosen_actions[1]] += REWARD_MATRIX.loc[chosen_actions[1],chosen_actions[0]]
        for j in range(k):
            agents_chosen_actions.loc[chosen_agents[j],(agents_chosen_number.loc[chosen_agents[j]]-1)%M] =chosen_actions[j]
        for j in range(k):
            if agents_chosen_number.loc[chosen_agents[j]] > M:
                n = agents_chosen_number.loc[chosen_agents[j]]
                los_ward = agents_chosen_reward.loc[chosen_agents[j],(n-1)%M]
                CRW.loc[chosen_agents[j],agents_chosen_actions.loc[chosen_agents[j],(n-1)%M]]-=los_ward
    print(CRW)
    print(agents_chosen_actions.loc[:,(M-5):M])
HCR()




