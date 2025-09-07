from torch.utils.data import Sampler, Dataset
import torch
import time 
from itertools import chain, accumulate


class Memory:  # collected from old policy
    def __init__(self):
        self.states = []
        self.truncation_values = []
        self.state_values = []
        self.actions = []
        self.rewards = []
        self.total_rewards = []
        self.is_terminals = []
        self.is_truncated = []
        self.logprobs = []
        self.episode_lengths = []
        self.total_steps = 0
        self.mean_std = 0
        self.timesteps = 0
        self.rollout_count = 0
        self.avg_length = 0
        self.speed = 0
        self.rmse = 0
        self.mean_error = 0
        self.img_captured = 0
        self.last_ids_or_values = {'states': 0, 'truncation_values': 0, 'state_values': 0, 'actions': 0, 'rewards': 0,
                                   'total_rewards': 0, 'is_terminals': 0, 'is_truncated': 0, 'logprobs': 0,
                                   'episode_lengths': 0,'total_steps': 0, 'mean_std': 0,
                                   'timesteps': 0, 'rollout_count': 0, 'avg_length': 0, 'speed': 0, 'rmse': 0,
                                   'mean_error': 0, 'img_captured': 0,
                                   'num_collisions': 0}

    def clear_memory(self):
        del self.states[:]
        del self.truncation_values[:]
        del self.state_values[:]
        del self.actions[:]
        del self.rewards[:]
        del self.total_rewards[:]
        del self.is_terminals[:]
        del self.is_truncated[:]
        del self.logprobs[:]
        del self.episode_lengths[:]
        del self.last_ids_or_values

        self.total_rewards = []
        self.total_steps = 0
        self.episode_lengths = []
        self.mean_std = 0
        self.timesteps = 0
        self.rollout_count = 0
        self.avg_length = 0
        self.speed = 0
        self.rmse = 0
        self.mean_error = 0
        self.img_captured = 0
        self.last_ids_or_values = {'states': 0, 'truncation_values': 0, 'state_values': 0, 'actions': 0, 'rewards': 0,
                                   'total_rewards': 0, 'is_terminals': 0, 'is_truncated': 0, 'logprobs': 0,
                                   'episode_lengths': 0,
                                   'total_steps': 0, 'mean_std': 0,
                                   'timesteps': 0, 'rollout_count': 0, 'avg_length': 0, 'speed': 0, 'rmse': 0,
                                   'mean_error': 0, 'img_captured': 0,
                                   'num_collisions': 0}

    def clear_running_episode(self):
        self.states = self.states[:self.last_ids_or_values['states']]
        self.truncation_values = self.truncation_values[:self.last_ids_or_values['truncation_values']]
        self.state_values = self.state_values[:self.last_ids_or_values['state_values']]
        self.actions = self.actions[:self.last_ids_or_values['actions']]
        self.rewards = self.rewards[:self.last_ids_or_values['rewards']]
        self.total_rewards = self.total_rewards[:self.last_ids_or_values['total_rewards']]
        self.is_terminals = self.is_terminals[:self.last_ids_or_values['is_terminals']]
        self.is_truncated = self.is_truncated[:self.last_ids_or_values['is_truncated']]
        self.logprobs = self.logprobs[:self.last_ids_or_values['logprobs']]
        self.episode_lengths = self.episode_lengths[:self.last_ids_or_values['episode_lengths']]
        self.total_steps = self.last_ids_or_values['total_steps']
        self.mean_std = self.last_ids_or_values['mean_std']
        self.timesteps = self.last_ids_or_values['timesteps']
        self.rollout_count = self.last_ids_or_values['rollout_count']
        self.avg_length = self.last_ids_or_values['avg_length']
        self.speed = self.last_ids_or_values['speed']
        self.rmse = self.last_ids_or_values['rmse']
        self.mean_error = self.last_ids_or_values['mean_error']
        self.img_captured = self.last_ids_or_values['img_captured']

    def checkpoint(self):
        self.last_ids_or_values['states'] = len(self.states)
        self.last_ids_or_values['truncation_values'] = len(self.truncation_values)
        self.last_ids_or_values['state_values'] = len(self.state_values)
        self.last_ids_or_values['actions'] = len(self.actions)
        self.last_ids_or_values['rewards'] = len(self.rewards)
        self.last_ids_or_values['total_rewards'] = len(self.total_rewards)
        self.last_ids_or_values['is_terminals'] = len(self.is_terminals)
        self.last_ids_or_values['is_truncated'] = len(self.is_truncated)
        self.last_ids_or_values['logprobs'] = len(self.logprobs)
        self.last_ids_or_values['episode_lengths'] = len(self.episode_lengths)
        self.last_ids_or_values['total_steps'] = self.total_steps
        self.last_ids_or_values['mean_std'] = self.mean_std
        self.last_ids_or_values['timesteps'] = self.timesteps
        self.last_ids_or_values['rollout_count'] = self.rollout_count
        self.last_ids_or_values['avg_length'] = self.avg_length
        self.last_ids_or_values['speed'] = self.speed
        self.last_ids_or_values['rmse'] = self.rmse
        self.last_ids_or_values['mean_error'] = self.mean_error
        self.last_ids_or_values['img_captured'] = self.img_captured

    def get_sequence(self, seq_length, id=-1, freq_fac=1):
        states = torch.stack(self.states[max(-(seq_length - 1) * freq_fac - 1, -len(self.states))::freq_fac], dim=1)
        attention_mask = torch.zeros((states.shape[0], states.shape[1]))
        if id == -1:
            return states, attention_mask
        else:
            return states[id], attention_mask[id]


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, memory, rewards):
        self.states = memory.states  # Keep as float16
        self.actions = memory.actions
        self.logprobs = memory.logprobs
        self.rewards = rewards
        self.is_truncated = memory.is_truncated
        self.is_terminal = memory.is_terminals


    def __len__(self):
        return len(self.is_terminal)

    def __getitem__(self, idx):

        # Collect list of tensors for this sequence
        state = self.states[idx]

        actions = self.actions[idx]
        logprob = self.logprobs[idx]
        reward = self.rewards[idx]
        is_truncated = self.is_truncated[idx]
        is_terminal = self.is_terminal[idx]

        return (state,
                actions,
                logprob,
                reward,
                is_truncated,
                is_terminal, idx)

def merge_memories(gen_memories):
    t1 = time.time()
    memory = Memory()
    add_ids = list(accumulate([len(this_memory.rewards) for this_memory in gen_memories]))
    for i, this_memory in enumerate(gen_memories[1:]):
        for j, (id, value) in enumerate(this_memory.truncation_values):
            this_memory.truncation_values[j] = (id + add_ids[i], value)
    memory.states = list(chain.from_iterable(memory.states for memory in gen_memories))
    memory.truncation_values = list(chain.from_iterable(memory.truncation_values for memory in gen_memories))
    memory.state_values = list(chain.from_iterable(memory.state_values for memory in gen_memories))
    memory.actions = list(chain.from_iterable(memory.actions for memory in gen_memories))
    memory.rewards = list(chain.from_iterable(memory.rewards for memory in gen_memories))
    memory.is_terminals = list(chain.from_iterable(memory.is_terminals for memory in gen_memories))
    memory.is_truncated = list(chain.from_iterable(memory.is_truncated for memory in gen_memories))
    memory.logprobs = list(chain.from_iterable(memory.logprobs for memory in gen_memories))
    memory.episode_lengths = list(chain.from_iterable(memory.episode_lengths for memory in gen_memories))
    num_workers = len(gen_memories)
    memory.rollout_count = sum(memory.rollout_count for memory in gen_memories)
    memory.avg_length = sum(memory.avg_length for memory in gen_memories) / num_workers
    memory.speed = sum(memory.speed for memory in gen_memories) / (memory.rollout_count)
    memory.rmse = sum(memory.rmse for memory in gen_memories) / (memory.rollout_count)
    memory.mean_error = sum(memory.mean_error for memory in gen_memories) / (memory.rollout_count)
    memory.timesteps = memory.avg_length * memory.rollout_count
    memory.mean_std = torch.stack([memory.mean_std for memory in gen_memories]).sum(dim=0) / memory.rollout_count
    t2 = time.time()
    print(t2 - t1)
    return memory

def merge_stats(gen_memories):
    memory = Memory()
    memory.rewards = list(chain(*[memory.rewards for memory in gen_memories]))
    print(f'First: {len(list(chain.from_iterable(memory.rewards for memory in gen_memories)))}')
    print(f'Second: {len(list(chain(*[memory.rewards for memory in gen_memories])))}')
    memory.is_terminals = list(chain(*[memory.is_terminals for memory in gen_memories]))
    memory.is_truncated = list(chain(*[memory.is_truncated for memory in gen_memories]))
    num_workers = len(gen_memories)
    memory.rollout_count = sum([memory.rollout_count for memory in gen_memories])
    print(f'Rollout Count: {memory.rollout_count}')
    memory.avg_length = sum([memory.avg_length for memory in gen_memories]) / num_workers
    memory.total_rewards = sum(list(chain(*[memory.total_rewards for memory in gen_memories]))) / (memory.rollout_count)
    memory.speed = sum([memory.speed for memory in gen_memories]) / (memory.rollout_count)
    memory.rmse = sum([memory.rmse for memory in gen_memories]) / (memory.rollout_count)
    memory.mean_error = sum([memory.mean_error for memory in gen_memories]) / (memory.rollout_count)
    memory.img_captured = sum([memory.img_captured for memory in gen_memories]) / (memory.rollout_count)
    memory.timesteps = memory.avg_length * memory.rollout_count
    memory.mean_std = torch.stack([memory.mean_std for memory in gen_memories]).sum(dim=0) / memory.rollout_count
    return memory
