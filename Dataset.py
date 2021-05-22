import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Batch_DataSet(torch.utils.data.Dataset):

    def __init__(self, obs, actions, adv, v_t, old_action_prob):
        super().__init__()
        self.obs = obs
        self.actions = actions
        self.adv = adv
        self.v_t = v_t
        self.old_action_prob = old_action_prob
        
    def __len__(self):
        return self.obs.shape[0]
    
    def __getitem__(self, i):
        return self.obs[i],self.actions[i],self.adv[i],self.v_t[i],self.old_action_prob[i]