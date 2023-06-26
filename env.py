import gym
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

random.seed(19990406)

class FraudDetectionEnv(gym.Env):
    def __init__(self, max_turns=30):
        '''
        Credit Card Fraud Detection(kaggle) 
        '''
        self.credit_card_dataset = './dataset/creditcard.csv'
        self.df_credit_card = pd.DataFrame(pd.read_csv(self.credit_card_dataset))
        
        '''
        simple version to deal with unbalanced dataset
        '''
        self.obs1 = self.df_credit_card[self.df_credit_card['Class']==1]
        self.obs0 = self.df_credit_card[self.df_credit_card['Class']==0]
        self.in_sample_0, self.out_of_sample_0 = train_test_split(self.obs0, test_size = 0.9,
                                                                  random_state = 19990406)
        self.train_0, self.test_0 = train_test_split(self.in_sample_0, test_size = 0.99,
                                                     random_state = 19990406)
        self.in_sample_1, self.out_of_sample_1 = train_test_split(self.obs1, test_size = 0.5,
                                                                  random_state = 19990406)
        self.train_1, self.test_1 = train_test_split(self.in_sample_1, test_size = 0.5,
                                                     random_state = 19990406)
        self.train = pd.concat([self.train_0,self.train_1],axis=0,ignore_index=True)
        self.train = self.train.sample(frac=1)
        self.test = pd.concat([self.test_0,self.test_1],axis=0,ignore_index=True)
        self.test = self.test.sample(frac=1)
        self.out_of_sample = pd.concat([self.out_of_sample_0,self.out_of_sample_1],
                                       axis=0,ignore_index=True)
        self.out_of_sample = self.out_of_sample.sample(frac=1)
        
        '''
        change the [gym.spaces] type 
        if the algorithm need Box space type then change
        the most easy way is to set it Discrete
        '''
        self.ACTION_LOOKUP = {0: 'not_fraud', 1: 'fraud'}
        # self.observation_space = gym.spaces.Discrete(len(self.df_credit_card.iloc[0,:-1].values))
        self.observation_space = gym.spaces.Box(low=-np.inf,high=np.inf,
                                                shape=(len(self.df_credit_card.iloc[0,:-1].values),))
        self.action_space = gym.spaces.Discrete(len(self.ACTION_LOOKUP))
        # self.action_space = gym.spaces.Box(low=0,high=1,shape=(1,))
        
        self.index_to_sample = [_ for _ in range(len(self.train))]
        self.sampled_index = []
        
        self.state_idx = self._get_state()
        # self.state_idx = 0
        self.done = False
        self.max_turns = max_turns
        self.turns = 0
        self.reward = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        self.state = None
    
    def label_for(self, state_idx):
        return self.train.iloc[state_idx,-1]
    
    def step(self, action):
        tpr,fpr,ac = None,None,None
        label_for_current_state = self.label_for(self.state_idx)
        # print(action)
        if action == label_for_current_state:
            self.reward = 1
        else:
            self.reward = -1
        '''
        if need to record the tpr and fpr
        '''
        # if round(action[0]) == label_for_current_state:
        #     self.reward = 1
        # else:
        #     self.reward = -1
        
        # if round(action[0]) == 1:
        #     if label_for_current_state == 1:
        #         self.true_positives += 1
        #         self.reward = 1
        #     else:
        #         self.false_positives += 1
        #         self.reward = -1
        # # elif round(action[0]) == 0:
        # else:
        #     if label_for_current_state == 0:
        #         self.true_negatives += 1
        #         self.reward = 1
        #     else:
        #         self.false_negatives += 1
        #         self.reward = -1
        # if (self.false_positives + self.true_negatives) != 0:
        #     fpr = self.false_positives / (self.false_positives + self.true_negatives)
        # elif (self.true_positives + self.false_negatives) != 0:
        #     tpr = self.true_positives / (self.true_positives + self.false_negatives)
        
        if (self.false_positives + self.true_negatives+self.true_positives + self.false_negatives)!=0:
            ac = (self.true_positives+self.false_negatives)/(self.false_positives +
                                                             self.true_negatives +
                                                             self.true_positives + 
                                                             self.false_negatives)
        info = {'false_positive_rate': fpr,
                'true_positive_rate': tpr,
                'accuracy_rate': ac}
        self.turns += 1
        if self.turns == self.max_turns:
            self.done = True
        self.state_idx = self._get_state()
        return self.train.iloc[self.state_idx,:-1].values, self.reward, self.done, info
    
    def reset(self):
        self.turns = 0
        self.done = False
        self.state_idx = self._get_state()
        self.reward = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        return self.train.iloc[self.state_idx,:-1].values
        
    def _get_state(self):
        '''
        if non-replacement sampling is needed
        '''
        # self.index_to_sample = [item for item in self.index_to_sample if item not in self.sampled_index]
        self.state_idx = random.sample(self.index_to_sample,1)
        # self.sampled_index.append(self.state_idx)
        return self.state_idx[0]