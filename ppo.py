"""PPO for path tracking in Carla"""

import os
import torch
import torch.nn as nn
from CarlaTrafficEnv import Carla_Traffic_Env
import numpy as np
import argparse
import pickle
import time
import ray
from Start_server import kill, start_server
import subprocess
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from Transformer_Memory import Memory, CustomDataset
from Vision_Policy import Policy
from visualize_training_data import generate_plots
from multiprocessing import shared_memory
import copy
from Transformer_Memory import merge_memories, merge_stats
from Image_Encoding import CompactCNN

parser = argparse.ArgumentParser(description='PyTorch PPO for continuous controlling')
parser.add_argument('--solved_reward', type=float, default=3000, help='stop training if avg_reward > solved_reward')
parser.add_argument('--print_interval', type=int, default=4, help='how many episodes to print the results out')
parser.add_argument('--save_interval', type=int, default=12, help='how many episodes to save a checkpoint')
parser.add_argument('--max_episodes', type=int, default=100000)
parser.add_argument('--max_timesteps', type=int, default=10 * 40)
parser.add_argument('--update_timesteps', type=int, default=8 * 10 * 40, help='how many timesteps to update the policy')
parser.add_argument('--batch_size', type=int, default=256, help='how many timesteps to update the policy')
parser.add_argument('--K_epochs', type=int, default=3, help='update the policy for how long time everytime')
parser.add_argument('--eps_clip', type=float, default=0.2, help='epsilon for p/q clipped')
parser.add_argument('--gamma', type=float, default=0.975, help='discount factor')
parser.add_argument('--lr', type=float,default=1e-5)
parser.add_argument('--ckpt_folder', default='./checkpoints', help='Location to save checkpoint models')
parser.add_argument('--restore', default=False, action='store_true', help='Restore and go on training?')
parser.add_argument('--cuda', default=False, help='Use GPU for Optimization?')
parser.add_argument('--num_workers', default=1, help='Number of parallel data generators')
opt = parser.parse_args()

# Hardware Settings
num_gpus = 0
ray.init(num_gpus=num_gpus)
if opt.cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Hyperparameters
scaler = GradScaler()
scaling_fac = 3
min_std = 0.05      #min standard deviation
grad_acc_iter = 1

# Distributional RL
distributed_critic = True
num_quantiles = 32
adv_quantile = 15

# Training performance lists
eval_list = []
rmse_list = []
mean_error_list = []
speed_list = []
mean_std_list = []
    
######################################### Am besten nichts ändern #######################################################################################################################################################

if opt.restore:
    with open(opt.ckpt_folder + "/policy_and_optimizer_data.pkl", "rb") as f:
        train_history = pickle.load(f)
    eval_list = train_history["Reward"]
    rmse_list = train_history["RMSE"]
    mean_error_list = train_history["Mean Error"]
    speed_list = train_history["Speed"]
    mean_std_list = train_history["Stds"]


@ray.remote
class StepCounter:
    def __init__(self): self.total = 0
    def add(self, n:int): self.total += int(n)
    def get(self): return self.total
    def reset(self): self.total = 0


@ray.remote(num_gpus=num_gpus/opt.num_workers)
class Trajectory_Generator:
    def __init__(self, policy, port, map, shm_name, counter):
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        self.stop_flag = np.ndarray((1,), dtype=np.bool_, buffer=existing_shm.buf)
        self.shm = existing_shm  # keep reference so it's not GC’ed

        if opt.cuda:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            # Take the first assigned GPU (if num_gpus=1, there's only one)
            global_gpu_id = int(visible_devices[0])
            subprocess.run(
                    f'taskset -c {int((port/1000-2)*6)}-{int((port/1000-2)*6+5)} ../carla_custom/carla_build/CarlaUE4.sh -carla-port={port} -vulkan -nosound -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter={global_gpu_id} -RenderOffScreen -quality-level=Low &',
                    shell=True) # 
            #start_server(port)
        else:
            start_server(port)
        time.sleep(15)

        self.memory = Memory()

        print('Attempt to connect')    

        # Policies
        if distributed_critic:
            self.policy = Policy(501+21, 2, min_std, scaling_fac, num_quantiles=num_quantiles)
        else:
            self.policy = Policy(501+21, 2, min_std, scaling_fac, num_quantiles=1)

        self.policy.set_policy(policy)

        if opt.cuda:
            self.policy = self.policy.cuda()

        image_encoder = CompactCNN(output_dim=501)#ImageEncoder(output_dim=245)#
        if opt.cuda:
            image_encoder = image_encoder.cuda()

        self.env = Carla_Traffic_Env(port=port, map = map,
                                     CNN=image_encoder, cuda = opt.cuda)

        self.counter = counter
        print('Connected to Server')

    def select_action(self, state, memory):
        action, state_value = self.policy.act(state, memory, det=False)
        return action.cpu().numpy().squeeze(), state_value

    def get_total_steps(self):
        return self.memory.total_steps


    def set_policy_weights(self, state_dict_ref):
        self.policy.set_state_dict(state_dict_ref)

    def clear_memory(self):
        self.memory.clear_memory()

    def get_stats(self):
        memory = Memory()
        memory.rewards = self.memory.rewards
        memory.is_terminals = self.memory.is_terminals
        memory.is_truncated = self.memory.is_truncated
        memory.total_rewards = self.memory.total_rewards
        memory.rollout_count = self.memory.rollout_count
        memory.avg_length = self.memory.avg_length
        memory.speed = self.memory.speed
        memory.rmse = self.memory.rmse
        memory.mean_error = self.memory.mean_error
        memory.timesteps = self.memory.timesteps
        memory.mean_std = self.memory.mean_std
        return memory

    def get_memory(self):
        return ray.put(self.memory)

    def set_policy(self, policy):
        self.policy = copy.deepcopy(policy).cuda()

    def collect_data(self):
        while True:
            torch.cuda.empty_cache()
            if self.stop_flag:
                break
            self.memory.rollout_count += 1

            # Reset statistics
            rmse = 0
            speed = 0
            mean_error = 0
            total_reward = 0
            self.policy.mean_std = 0
            self.policy.timestep = 1
            state, all_stats = self.env.reset()

            dist = all_stats[:, 1]
            velocity = all_stats[:, 2]
            speed += velocity * 3.6
            rmse += dist ** 2
            mean_error += dist

            crashed = False

            for t in range(opt.max_timesteps):
                if self.stop_flag:
                    break
                actions, state_values = self.select_action(state,  self.memory)
                state, reward, done, all_stats, crashed = self.env.step(np.clip(actions / scaling_fac, -1, 1))
                
                if crashed:
                    break

                total_reward += reward
                dist = all_stats[:, 1]
                velocity = all_stats[:, 2]
                speed += velocity * 3.6
                rmse += dist ** 2
                mean_error += dist

                # Save Data
                self.memory.rewards.append(reward)
                self.memory.state_values.append(state_values)
                self.memory.is_terminals.append(done)
                self.memory.is_truncated.append(t == opt.max_timesteps - 1 and not done)

                # If Episode is truncated save the final image and observations
                if t == opt.max_timesteps - 1 and not done:
                    with autocast('cuda'):
                        state_value = self.policy.eval_truncation(state)
                    self.memory.truncation_values.append(
                        (len(self.memory.rewards), state_value.detach().to('cpu')))
                    break
                elif done:
                    break

            if self.stop_flag:
                self.memory.clear_running_episode()
                break

            if crashed:
                self.memory.clear_running_episode()
                continue

            self.env.cleanup()
            self.memory.episode_lengths.append(t)
            self.memory.total_steps += t
            self.counter.add.remote(t)
            self.memory.avg_length *= self.memory.rollout_count - 1
            self.memory.avg_length += t
            self.memory.avg_length /= self.memory.rollout_count
            self.memory.rmse += np.mean(np.sqrt(rmse / (t + 1)))
            self.memory.total_rewards.append(np.mean(total_reward))
            self.memory.timesteps += t + 1
            self.memory.mean_error += np.mean(mean_error / (t + 1))
            self.memory.speed += np.mean(speed / (t + 1))
            #print( self.policy.mean_std)
            self.memory.mean_std += self.policy.mean_std
            self.memory.checkpoint()


class PPO:
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, restore=False, ckpt=None, advantage_normalization=True, grad_clip_param=-1):

        # Algorithm settings
        self.advantage_normalization = advantage_normalization

        self.grad_clip_param = grad_clip_param

        # Set Hyperparameters
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # Initialize and potentially reload policy and optimizer
        if distributed_critic:
            self.policy = Policy(501+21, 2, min_std, scaling_fac, num_quantiles=num_quantiles)
        else:
            self.policy = Policy(501+21, 2, min_std, scaling_fac, num_quantiles=1)

        if restore:
            pretrained_model = torch.load(ckpt, map_location=torch.device('cuda:0'))
            self.policy.set_state_dict(pretrained_model['policy_state_dict'])

        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr, betas=betas)
        if restore:
            pretrained_model = torch.load(ckpt, map_location=torch.device('cuda:0'))
            self.optimizer.load_state_dict(pretrained_model['optimizer_state_dict'])

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                param_group['betas'] = betas

            if opt.cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to("cuda:0")

        if opt.cuda:
            self.policy=self.policy.cuda()

        # Loss functions
        self.MSE_loss = nn.MSELoss() # Alternatively: SmoothL1Loss()

    def precompute_gae(self, memory, lamb=.95):  # Previously: lamb = .95
        advantages = []
        rewards = [torch.as_tensor(x).to(device) for x in memory.rewards]

        with torch.no_grad():
            truncation_counter = 0
            if len(memory.truncation_values) > 0:
                if distributed_critic:
                    truncation_values = [torch.mean(x[1], dim=1).to(device) for x in memory.truncation_values]
                else:
                    truncation_values = [x[1].to(device) for x in memory.truncation_values]
            next_state_value = 0
            for t in range(len(rewards) - 1, -1, -1):
                if distributed_critic:
                    state_value = torch.mean(memory.state_values[t], dim=1).to(device)
                else:
                    state_value = memory.state_values[t].to(device)  # .cuda()
                if memory.is_truncated[t]:
                    truncation_counter += 1
                    advantages.insert(0, (rewards[t] + self.gamma * truncation_values[-truncation_counter] - state_value).reshape((-1,)))  # .cuda()
                elif memory.is_terminals[t]:
                    advantages.insert(0, (rewards[t].reshape((-1, 1)) - state_value).reshape((-1,)))
                else:
                    advantages.insert(0, (rewards[t] + self.gamma * next_state_value - state_value + self.gamma * lamb * advantages[0]).reshape((-1,)))

                next_state_value = state_value.clone()

        return advantages

    def compute_discounted_rewards(self, memory):
        rewards = []

        with torch.no_grad():
            # Estimate/Compute cumulated rewards
            for i, (reward, is_terminal) in enumerate(zip(reversed(memory.rewards), reversed(memory.is_terminals))):
                if memory.is_truncated[-(i + 1)] and not is_terminal:
                    if distributed_critic:
                        truncation_value = np.array([torch.mean(truncation_value[1], dim=1) for truncation_value in memory.truncation_values if truncation_value[0] == len(memory.rewards) - i])
                    else:
                        truncation_value = np.array([truncation_value[1] for truncation_value in memory.truncation_values if truncation_value[0] == len(memory.rewards) - i])
                    discounted_reward = truncation_value
                elif is_terminal:
                    discounted_reward = 0
                discounted_reward = reward.reshape((-1, 1)) + self.gamma * discounted_reward
                rewards.insert(0, torch.from_numpy(discounted_reward))
        return rewards

    def quantile_huber_loss(self, pred, target, taus, kappa=1.0):
        """
        pred:   Tensor (n, N) — Vorhergesagte Quantile pro Beispiel
        target: Tensor (n,)   — Zielwerte (z.B. stochastisch gesampelte Returns)
        taus:   Tensor (N,)   — Quantile pro Output (z.B. [1/33, 2/33, ..., 32/33])
        kappa:  float         — Huber-Grenze

        Gibt: Skalarischen loss zurück.
        """
        n, N = pred.shape

        # (n, 1) → (n, N): Broadcast target zu jedem Quantil
        target = target.unsqueeze(1).expand(n, N)

        # TD-Fehler (delta): (n, N)
        u = target - pred

        # Huber loss: Stückweise definiert
        huber = torch.where(
            u.abs() <= kappa,
            0.5 * u.pow(2),
            kappa * (u.abs() - 0.5 * kappa)
        )

        # Gewichtung je nach Quantil-Level
        # (tau - 𝟙{u < 0}) * huber
        # shape: (n, N)
        loss = (taus.unsqueeze(0) - (u.detach() < 0).float()).abs() * huber / kappa

        return loss.mean()  # sum(dim=1).mean()  # Mittelwert über Batch

    # Deprecated
    def update(self, memory):
        def get_total_grad_norm(model, norm_type=2):
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
            total_norm = total_norm ** (1. / norm_type)
            return total_norm

        rewards = self.compute_discounted_rewards(memory)

        t1 = time.time()
        advantages = self.precompute_gae(memory, lamb=.95)  # Shape (num_agents,total timesteps)
        t2 = time.time()
        print(f'Pre-Computation time: {t2 - t1}')
        t1 = time.time()

        quantiles = torch.tensor([(k + 1) / (num_quantiles + 1) for k in range(num_quantiles)]).cuda()
        dataset = CustomDataset(memory, rewards)
        dataloader = DataLoader(
            dataset,
            batch_size=opt.batch_size,  # Tune batch size for performance
            drop_last=True,
            num_workers=0,
            shuffle=True,
            pin_memory=False  # Speeds up CPU → GPU transfers
        )

        t2 = time.time()
        print(f'Valid_indices and Dataloader: {t2 - t1}')

        # Train policy for K epochs:
        for i in range(self.K_epochs):
            for j, batch in enumerate(dataloader):

                (old_states, old_actions, old_logprobs, batch_rewards, is_truncated, is_terminal, indices) = batch

                old_states = old_states.to(device)
                old_actions = old_actions.to(device)
                old_logprobs = old_logprobs.to(device)
                batch_rewards = batch_rewards.to(device)

                with autocast('cuda'):

                    logprobs, state_values, dist_entropy = self.policy(old_states, old_actions)

                    # Importance ratio: p/q
                    ratios = torch.exp(logprobs - old_logprobs.detach().squeeze())

                    #if torch.any(ratios>5):
                    #    continue
                    
                    # Batch advantages & normalization
                    batch_advantages = torch.cat([(advantages[t]) for t in indices], dim=0)
                    if opt.cuda:
                        batch_advantages = batch_advantages.cuda()
                    if self.advantage_normalization:
                        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                    
                    # Actor loss using Surrogate loss
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                    actor_loss = - torch.min(surr1, surr2)

                    # Critic loss: critic loss - entropy
                   
                    if distributed_critic:
                        critic_loss = 3 * self.quantile_huber_loss(state_values, batch_rewards.squeeze(),quantiles).cuda()
                    else:
                        critic_loss = 0.5 * self.MSE_loss(batch_rewards, state_values)

                    entropy_loss = - 0.02 * dist_entropy

                    
                    print('Max. importance sampling ratio: ' + str(np.round(torch.max(ratios).item(), decimals=3)))

                    if 0 in indices:
                        if distributed_critic:
                            print('State_values of sample 0: ' + str(state_values[0, :]))
                            print('Max_value: ' + str(
                                torch.max(state_values[:, num_quantiles - 1]).item()) + ', Max_Reward: ' + str(
                                torch.max(batch_rewards).item()))
                            print(
                                'Min_value: ' + str(torch.min(state_values[:, 0]).item()) + ', Min_Reward: ' + str(
                                    torch.min(batch_rewards).item()))
                        else:
                            print('Max_value: ' + str(torch.max(state_values).item()) + ', Max_Reward: ' + str(
                                torch.max(batch_rewards).item()))
                            print('Min_value: ' + str(torch.min(state_values).item()) + ', Min_Reward: ' + str(
                                torch.min(batch_rewards).item()))

                        if i == 0:
                            print('Initial critic loss: ' + str(critic_loss.item()))
                            print('Initial actor loss: ' + str(actor_loss.mean().item()))
                            print('Initial entropy loss: ' + str(entropy_loss.mean().item()))
                            
                        elif i == self.K_epochs - 1:
                            print('Final critic loss: ' + str(critic_loss.item()) + ', Final max ratio: ' + str(
                                torch.max(ratios).item()))
                            print('Final actor loss: ' + str(actor_loss.mean().item()))
                            print('Final entropy loss: ' + str(entropy_loss.mean().item()))

                    # Total loss
                    loss = (critic_loss + actor_loss + entropy_loss).mean() / grad_acc_iter # + messaging_loss.mean(dim=1)

                # Backward gradients
                scaler.scale(loss).backward()

                if j % grad_acc_iter == grad_acc_iter - 1:
                    if self.grad_clip_param > 0:
                        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.grad_clip_param)

                    grad_norm = get_total_grad_norm(self.policy.model)
                    print(f"Gradient norm: {grad_norm:.4f}")
                    scaler.step(self.optimizer)  # Apply gradients
                    scaler.update()

                    # Clear accumulated gradients
                    self.optimizer.zero_grad()

        # Copy new weights to old_policy
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()


def train(state_dim, action_dim, solved_reward,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, restore, print_interval=10, save_interval=100, cuda=False,
          verbose=False):

    ckpt = ckpt_folder + '/policy_and_optimizer.pth'
    if restore:
        print('Load checkpoint from {}'.format(ckpt))

    ppo = PPO(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, restore=restore, ckpt=ckpt, grad_clip_param=5000000)

    shm = shared_memory.SharedMemory(create=True, size=1)
    stop_flag = np.ndarray((1,), dtype=np.bool_, buffer=shm.buf)
    stop_flag[0] = False
    counter = StepCounter.remote()

    generators = []
    kill()
    for i in range(opt.num_workers):
        port = 2000 + i * 1000
        generators.append(Trajectory_Generator.remote(ppo.policy, port, 'Town01_Opt', shm.name, counter))
        #generators.append(Trajectory_Generator(ppo.policy, port, 'Town01_Opt', shm.name, counter))
        time.sleep(20)

    for update_steps in range(1, max_episodes + 1):
        # start workers asynchronously
        t1 = time.time()
        futures = [gen.collect_data.remote() for gen in generators]
        #generators[0].collect_data()
        # monitor progress
        while True:
            time.sleep(.5)
            total_steps = ray.get(counter.get.remote())
            #print(total_steps)
            if total_steps >= update_timestep:
                stop_flag[0] = True
                break

        # wait for all workers to finish 
        results = ray.get(futures)

        # Reset stop_flags
        stop_flag[0] = False
        counter.reset.remote()

        
        t2 = time.time()

        timesteps = sum(ray.get([gen.get_total_steps.remote() for gen in generators]))
        print(f'Timesteps: {timesteps}')
        print(f'Generated data for {(t2 - t1)}s')

        t1=time.time()
        stats_memories = ray.get([gen.get_stats.remote() for gen in generators])
        print(stats_memories)
        memory = merge_stats(stats_memories)
        print(f'Merge Stats Time:  {time.time()-t1}')
        eval_list.append(memory.total_rewards)
        rmse_list.append(memory.rmse)
        mean_error_list.append(memory.mean_error)
        speed_list.append(memory.speed)
        mean_std_list.append(memory.mean_std.cpu().numpy())

        print(
            'Data generation round: {} \t Avg length: {} \t Est avg reward: {}, \t Avg. RMSE: {}, \t Avg. Mean Error: {}, \t Avg. Speed: {}, Avg. Std: {}'.format(
                update_steps, np.round(memory.avg_length, decimals=3),
                np.round(memory.total_rewards, decimals=3),
                np.round(memory.rmse, decimals=3),
                np.round(memory.mean_error, decimals=3),
                np.round(memory.speed, decimals=3),
                np.round(memory.mean_std.cpu(), decimals=3)))

        print('Update!')

        t1 = time.time()
        memory = merge_memories(ray.get(ray.get([gen.get_memory.remote() for gen in generators])))
        print(f'Merged Memory:  {time.time() - t1}s')
        t1 = time.time()
        ppo.update(memory)
        print(f'Time to update: {time.time() - t1}s')
        weights_ref = ray.put(ppo.policy.model.state_dict())
        ray.get([gen.set_policy_weights.remote(weights_ref) for gen in generators])
        torch.save({
            'policy_state_dict': ppo.policy.model.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict()
        }, ckpt_folder + '/policy_and_optimizer.pth')
        with open(ckpt_folder + '/policy_and_optimizer_data.pkl', "wb") as f:
            pickle.dump(
                {'Reward': eval_list, 'RMSE': rmse_list, 'Mean Error': mean_error_list, 'Speed': speed_list,
                     'Stds': mean_std_list}, f)
        generate_plots()
        ray.get([gen.clear_memory.remote() for gen in generators])
        print('Save a checkpoint!')

if __name__ == '__main__':
    if not os.path.exists(opt.ckpt_folder):
        os.mkdir(opt.ckpt_folder)
    print(
        "__________________________________________________________________________________________________________________________________________________")
    env_name = 'Traffic Environment'

    state_dim = 11
    action_dim = 2
    print('Environment: {}\nState Size: {}\nAction Size: {}\n'.format(env_name, state_dim, action_dim))

    train(state_dim, action_dim, solved_reward=opt.solved_reward,
          max_episodes=opt.max_episodes, max_timesteps=opt.max_timesteps, update_timestep=opt.update_timesteps,
          K_epochs=opt.K_epochs, eps_clip=opt.eps_clip, gamma=opt.gamma, lr=opt.lr, betas=(0.9, 0.990),
          ckpt_folder=opt.ckpt_folder, restore=opt.restore, print_interval=opt.print_interval,
          save_interval=opt.save_interval, cuda=opt.cuda)




