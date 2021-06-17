import numpy as np
import argparse
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Agent import PPO_Agent
from Atari_Wrapper import Atari_Wrapper
import Environment_Runner as runner
from Dataset import Batch_DataSet

device = torch.device("cuda:0")
dtype = torch.float

def compute_advantage_and_value_targets(rewards, values, dones, gamma, lam):
    
    advantage_values = []
    old_adv_t = torch.tensor(0.0).to(device)
    
    value_targets = []
    old_value_target = values[-1]
    
    for t in reversed(range(len(rewards)-1)):
        
        if dones[t]:
            old_adv_t = torch.tensor(0.0).to(device)
        
        # ADV
        delta_t = rewards[t] + (gamma*(values[t+1])*int(not dones[t+1])) - values[t]
        
        A_t = delta_t + gamma*lam*old_adv_t
        advantage_values.append(A_t[0])
        
        old_adv_t = delta_t + gamma*lam*old_adv_t
        
        # VALUE TARGET
        value_target = rewards[t] + gamma*old_value_target*int(not dones[t+1])
        value_targets.append(value_target[0])
        
        old_value_target = value_target
    
    advantage_values.reverse()
    value_targets.reverse()
    
    return advantage_values, value_targets


def train(args):  
    
    # create folder to save networks, csv, hyperparameter
    folder_name = time.asctime(time.gmtime()).replace(" ","_").replace(":","_")
    os.mkdir(folder_name)
    
    # save the hyperparameters in a file
    f = open(f'{folder_name}/args.txt','w')
    for i in args.__dict__:
        f.write(f'{i},{args.__dict__[i]}\n')
    f.close()
    
    # arguments
    env_name = args.env
    num_stacked_frames = args.stacked_frames
    start_lr = args.lr 
    gamma = args.gamma
    lam = args.lam
    minibatch_size = args.minibatch_size
    T = args.T
    c1 = args.c1
    c2 = args.c2
    actors = args.actors
    start_eps = args.eps
    epochs = args.epochs
    total_steps = args.total_steps
    save_model_steps = args.save_model_steps

    # init
    
    # in/output    
    in_channels = num_stacked_frames
    num_actions = gym.make(env_name).env.action_space.n

    # network and optim
    agent = PPO_Agent(in_channels, num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=start_lr)
    
    # actors
    env_runners = []
    for actor in range(actors):

        raw_env = gym.make(env_name)
        env = Atari_Wrapper(raw_env, env_name, num_stacked_frames, use_add_done=args.lives)
        
        env_runners.append(runner.Env_Runner(env, agent, folder_name))
        
    num_model_updates = 0

    start_time = time.time()
    while runner.cur_step < total_steps:
        
        # change lr and eps over time
        alpha = 1 - (runner.cur_step / total_steps)
        current_lr = start_lr * alpha
        current_eps = start_eps * alpha
        
        #set lr
        for g in optimizer.param_groups:
            g['lr'] = current_lr
        
        # get data
        batch_obs, batch_actions, batch_adv, batch_v_t, batch_old_action_prob = None, None, None, None, None
    
        for env_runner in env_runners:
            obs, actions, rewards, dones, values, old_action_prob = env_runner.run(T)
            adv, v_t = compute_advantage_and_value_targets(rewards, values, dones, gamma, lam)
    
            # assemble data from the different runners 
            batch_obs = torch.stack(obs[:-1]) if batch_obs == None else torch.cat([batch_obs,torch.stack(obs[:-1])])
            batch_actions = np.stack(actions[:-1]) if batch_actions is None else np.concatenate([batch_actions,np.stack(actions[:-1])])
            batch_adv = torch.stack(adv) if batch_adv == None else torch.cat([batch_adv,torch.stack(adv)])
            batch_v_t = torch.stack(v_t) if batch_v_t == None else torch.cat([batch_v_t,torch.stack(v_t)]) 
            batch_old_action_prob = torch.stack(old_action_prob[:-1]) if batch_old_action_prob == None else torch.cat([batch_old_action_prob,torch.stack(old_action_prob[:-1])])
    
        # load into dataset/loader
        dataset = Batch_DataSet(batch_obs,batch_actions,batch_adv,batch_v_t,batch_old_action_prob)
        dataloader = DataLoader(dataset, batch_size=minibatch_size, num_workers=0, shuffle=True)
        
        
        # update
        for epoch in range(epochs):
             
            # sample minibatches
            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                if i >= 8:
                    break
                
                # get data
                obs, actions, adv, v_target, old_action_prob = batch 
                
                adv = adv.squeeze(1)
                # normalize adv values
                adv = ( adv - torch.mean(adv) ) / ( torch.std(adv) + 1e-8)
                
                # get policy actions probs for prob ratio & value prediction
                policy, v = agent(obs)
                # get the correct policy actions
                pi = policy[range(minibatch_size),actions.long()]
                
                # probaility ratio r_t(theta)
                probability_ratio = pi / (old_action_prob + 1e-8)
                
                # compute CPI
                CPI = probability_ratio * adv
                # compute clip*A_t
                clip = torch.clamp(probability_ratio,1-current_eps,1+current_eps) * adv     
                
                # policy loss | take minimum
                L_CLIP = torch.mean(torch.min(CPI, clip))
                
                # value loss | mse
                L_VF = torch.mean(torch.pow(v - v_target,2))
                
                # policy entropy loss 
                S = torch.mean( - torch.sum(policy * torch.log(policy + 1e-8),dim=1))

                loss = - L_CLIP + c1 * L_VF - c2 * S
                loss.backward()
                optimizer.step()
        
            
        num_model_updates += 1
         
        # print time
        if runner.cur_step%50000 < T*actors:
            end_time = time.time()
            print(f'*** total steps: {runner.cur_step} | time(50K): {end_time - start_time} ***')
            start_time = time.time()
        
        # save the network after some time
        if runner.cur_step%save_model_steps < T*actors:
            torch.save(agent,f'{folder_name}/{env_name}-{runner.cur_step}.pt')

    env.close()
    
if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    
    # set hyperparameter
    
    args.add_argument('-lr', type=float, default=2.5e-4)
    args.add_argument('-env', default='SeaquestNoFrameskip-v4')
    args.add_argument('-lives', type=bool, default=True)
    args.add_argument('-stacked_frames', type=int, default=4)
    args.add_argument('-gamma', type=float, default=0.99)
    args.add_argument('-lam', type=float, default=0.95)
    args.add_argument('-eps', type=float, default=0.1)
    args.add_argument('-c1', type=float, default=1.0)
    args.add_argument('-c2', type=float, default=0.01)
    args.add_argument('-minibatch_size', type=int, default=32)
    args.add_argument('-actors', type=int, default=8)
    args.add_argument('-T', type=int, default=129)
    args.add_argument('-epochs', type=int, default=3)
    args.add_argument('-total_steps', type=int, default=10000000)
    args.add_argument('-save_model_steps', type=int, default=1000000)
    args.add_argument('-report', type=int, default=50000)
    
    train(args.parse_args())
