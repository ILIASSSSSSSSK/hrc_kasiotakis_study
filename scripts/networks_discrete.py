import os
import rospy
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
https://github.com/EveLIn3/Discrete_SAC_LunarLander/blob/master/sac_discrete.py
"""
def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


# why retain graph? Do not auto free memory for one loss when computing multiple loss
# https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
def update_params(optim, loss):
    optim.zero_grad()
    loss.backward(retain_graph=True)
    optim.step()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class ReplayBuffer:
    """
    Convert to numpy
    """
    def __init__(self, memory_size):
        self.storage = []
        self.memory_size = memory_size
        self.next_idx = 0

    # add the samples
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        # get the next idx
        self.next_idx = (self.next_idx + 1) % self.memory_size

    # encode samples
    def _encode_sample(self, idx):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storage[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)

    # sample from the memory
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
        # Save Buffer       #changed this function in order to save experts data in different files after game ends
    def save_buffer(self, path):
        filename = "buffer_data.npy"
        file_path = os.path.join(path, filename)

        # Check if the file already exists
        if os.path.exists(file_path):
            i = 1
            while True:
                new_filename = "buffer_data_{}.npy".format(i)
                new_file_path = os.path.join(path, new_filename)
                if not os.path.exists(new_file_path):
                    filename = new_filename
                    break
                i += 1

        np.save(os.path.join(path, filename), self.storage)

    # Load Buffer
    def load_buffer(self, path):
        self.storage = np.load(path, allow_pickle=True).tolist()


class Dual_ReplayBuffer:
    """
    Convert to numpy
    """
    def __init__(self, memory_size,demo_data_path,percentages):
        #print(type(replay_buffer))
        #print(type(replay_buffer[0]))

        self.storage = []
        self.memory_size = memory_size
        self.next_idx = 0
        #self.expert_storage = replay_buffer.storage

        self.expert_storage = np.load(demo_data_path, allow_pickle=True).tolist()

        self.percentages = percentages

    # add the samples
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        # get the next idx
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def get_size(self):
        return len(self.storage)

    # encode samples
    def _encode_sample(self, idx,idx_exp):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storage[i]
            obs, action, reward, obs_, done= data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)

        for i in idx_exp:
            #print(self.expert_storage)

            data = self.expert_storage[i]
            #print("############",type(data))
            #print(len(data), data)

            obs, action, reward, obs_, done= data

            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)


        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)

    # sample from the memory
    def sample(self, batch_size, episode_number):
        print(self.percentages[episode_number])
        idx_exp = [random.randint(0, len(self.expert_storage) - 1) for _ in range(int(batch_size*self.percentages[episode_number]))]
        idx = [random.randint(0, len(self.storage) - 1) for _ in range(int(batch_size*(1-self.percentages[episode_number])))]
        return self._encode_sample(idx,idx_exp)
    # Save Buffer
    def save_buffer(self, path):
        np.save(path, self.storage)

    # Load Buffer
    def load_buffer(self, path):
        self.storage = np.load(path, allow_pickle=True).tolist()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units, name='actor', chkpt_dir='tmp/sac'):
        super(Actor, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        # os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.chdir(os.path.expanduser('~'))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.name = name
        self.actor_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, action_dim)
        ).apply(init_weights)

    def forward(self, s):
        actions_logits = self.actor_mlp(s)
        return F.softmax(actions_logits, dim=-1)

    def greedy_act(self, s):  # no softmax more efficient
        s = torch.from_numpy(s).float().to(device)
        actions_logits = self.actor_mlp(s)
        greedy_actions = torch.argmax(actions_logits, dim=-1, keepdim=True)
        return greedy_actions.item()

    def sample_act(self, s):
        s = torch.from_numpy(s).float().to(device)
        actions_logits = self.actor_mlp(s)
        actions_probs = F.softmax(actions_logits, dim=-1)
        actions_distribution = Categorical(actions_probs)
        action = actions_distribution.sample()
        return action.item()

    def save_checkpoint(self):
        if not rospy.get_param("/rl_control/Game/save_init_ag",False):  
            torch.save(self.state_dict(), self.checkpoint_file+"_init_agent")
        else:
            torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("I load weights")
        if rospy.get_param("/rl_control/Game/initialized_agent",False):
            if rospy.get_param("/rl_control/Game/lfd_participant_gameplay",False):
                self.load_state_dict(torch.load(rospy.get_param("/rl_control/Game/lfd_initialized_agent_dir")+rospy.get_param("/rl_control/Game/actor_name")))
                
            else:
                self.load_state_dict(torch.load(rospy.get_param("/rl_control/Game/initialized_agent_dir")+rospy.get_param("/rl_control/Game/actor_name")))
                
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units, name='critic', chkpt_dir='tmp/sac'):
        super(Critic, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        if chkpt_dir is not None:
            self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
            # os.makedirs(self.checkpoint_dir, exist_ok=True)
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

        self.qnet1 = DuelQNet(state_dim, action_dim, n_hidden_units)
        self.qnet2 = DuelQNet(state_dim, action_dim, n_hidden_units)

    def forward(self, s):  # S: N x F(state_dim) -> Q: N x A(action_dim) Q(s,a)
        q1 = self.qnet1(s)
        q2 = self.qnet2(s)
        return q1, q2

    def save_checkpoint(self):
        if not rospy.get_param("/rl_control/Game/save_init_ag",False):  
            torch.save(self.state_dict(), self.checkpoint_file+"_init_agent")
        else:
            torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if rospy.get_param("/rl_control/Game/initialized_agent",False):
            if rospy.get_param("/rl_control/Game/lfd_participant_gameplay",False):
                self.load_state_dict(torch.load(rospy.get_param("/rl_control/Game/lfd_initialized_agent_dir")+rospy.get_param("/rl_control/Game/critic_name")))
            else:
                self.load_state_dict(torch.load(rospy.get_param("/rl_control/Game/initialized_agent_dir")+rospy.get_param("/rl_control/Game/critic_name")))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))


class DuelQNet(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units, name='DuelQNet', chkpt_dir='tmp/sac'):
        super(DuelQNet, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        if chkpt_dir is not None:
            self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
            # os.makedirs(self.checkpoint_dir, exist_ok=True)
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        ).apply(init_weights)

        # self.q_head = nn.Linear(n_hidden_units, action_dim)

        self.action_head = nn.Linear(n_hidden_units, action_dim).apply(init_weights)
        self.value_head = nn.Linear(n_hidden_units, 1).apply(init_weights)

    def forward(self, s):
        s = self.shared_mlp(s)
        a = self.action_head(s)
        v = self.value_head(s)
        return v + a - a.mean(1, keepdim=True)
        # return self.q_head(s)
