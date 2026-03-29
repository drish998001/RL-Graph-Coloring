import os
import logging
import random
import numpy as np
import networkx as nx
import argparse
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import dgl
from ppo.framework_new import ProxPolicyOptimFramework
from ppo.actor_critic_new import ActorCritic
from ppo.graph_net_new import PolicyGraphConvNet, ValueGraphConvNet
from ppo.storage_new import RolloutStorage

from data.graph_dataset1 import get_er_15_20_dataset
from data.util1 import write_nx_to_metis

from env_new import ChannelAllocationEnv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir", 
    help="directory to store validation and test datasets",
    type=str,
    required = True
    )
parser.add_argument(
    "--device",
    help="id of gpu device to use, or 'cpu'",
    default="cpu"
)
args = parser.parse_args()
if args.data_dir is None:
    raise ValueError("You must specify --data-dir (e.g., --data-dir ./data)")
if str(args.device).isdigit():
    device = torch.device(f"cuda:{args.device}")
else:
    device = torch.device(args.device)


#device = torch.device(args.device)
base_data_dir = os.path.join(args.data_dir, "er_15_20")

# env
number_of_channels = 4

# actor critic
num_layers = 4
input_dim = 2
output_dim = number_of_channels + 2  
hidden_dim = 128

# optimization
init_lr = 1e-4
max_epi_t = 32
max_rollout_t = 32
max_update_t = 10000

# ppo
gamma = 1.0
clip_value = 0.2
optim_num_samples = 4
critic_loss_coef = 0.5 
reg_coef = 0.1
max_grad_norm = 0.5

# logging
vali_freq = 5
log_freq = 1

# dataset specific
dataset = "synthetic"
graph_type = "er"
min_num_nodes = 15
max_num_nodes = 20

# main
rollout_batch_size = 32
eval_batch_size = 1000
optim_batch_size = 16
train_num_samples = 1
eval_num_samples = 1

# initial values
best_vali_colored = -1e5
best_vali_zero = 1e5

# generate and save datasets
num_eval_graphs = 100

for mode in ["vali", "test"]:
    # make folder for storing graphs
    data_dir = os.path.join(base_data_dir, mode)
    os.makedirs(data_dir, exist_ok = True)
    print("Generating {} dataset at {}...".format(mode, data_dir))
    for g_idx in tqdm(range(num_eval_graphs)):
        nx_g_path = os.path.join(data_dir, "{:06d}.METIS".format(g_idx))

        # number of nodes in the graph is sampled uniformly at random
        num_nodes = random.randint(min_num_nodes, max_num_nodes)

        # make an ER graph from the networkX package
        nx_g = nx.erdos_renyi_graph(num_nodes, p = 0.15)

        # save the graph to METIS graph format
        write_nx_to_metis(nx_g, nx_g_path)

datasets = {
    "train": get_er_15_20_dataset("train"),
    "vali": get_er_15_20_dataset("vali", os.path.join(base_data_dir, "vali")),
    "test": get_er_15_20_dataset("test", os.path.join(base_data_dir, "test"))
}
    

# construct data loaders
def collate_fn(graphs):
    return dgl.batch(graphs)

data_loaders = {
    "train": DataLoader(
        datasets["train"],
        batch_size = rollout_batch_size,
        shuffle = True,
        collate_fn = collate_fn,
        num_workers = 0,
        drop_last = True
        ),
    "vali": DataLoader(
        datasets["vali"],
        batch_size = eval_batch_size,
        shuffle = False,
        collate_fn = collate_fn,
        num_workers = 0
        ),
    "test": DataLoader(  
        datasets["test"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
        )
        }

# construct environment
env = ChannelAllocationEnv(
    max_epi_t = max_epi_t,
    k = number_of_channels,  
    max_num_nodes = max_num_nodes,
    device = device
)

# construct rollout storage
rollout = RolloutStorage(
    max_t = max_rollout_t, 
    batch_size = rollout_batch_size, 
    num_samples = train_num_samples 
    )

# construct actor critic network
actor_critic = ActorCritic(
    actor_class = PolicyGraphConvNet,
    critic_class = ValueGraphConvNet,
    k = number_of_channels, 
    max_num_nodes = max_num_nodes, 
    hidden_dim = hidden_dim,
    num_layers = num_layers,
    device = device
    )

# construct PPO framework
framework = ProxPolicyOptimFramework(
    actor_critic = actor_critic,
    init_lr = init_lr,
    clip_value = clip_value,
    k = number_of_channels, 
    optim_num_samples = optim_num_samples,
    optim_batch_size = optim_batch_size,
    critic_loss_coef = critic_loss_coef, 
    reg_coef = reg_coef, 
    max_grad_norm = max_grad_norm, 
    device = device
    )    

# define evaluate function
def evaluate(mode, actor_critic):
    actor_critic.eval()
    cum_cnt = 0
    cum_eval_colored = 0.0
    cum_eval_zero = 0.0
    for g in data_loaders[mode]:
        g.set_n_initializer(dgl.init.zero_initializer)
        ob = env.register(g, num_samples = eval_num_samples)
        while True:
            with torch.no_grad():
                action = actor_critic.act(ob, g)

            ob, reward, done, info = env.step(action)
            if torch.all(done).item():
                cum_eval_zero += info['zero'].max(dim = 1)[0].sum().cpu()
                cum_eval_colored += info["colored"].max(dim = 1)[0].sum().cpu()
                cum_cnt += g.batch_size
                break
    actor_critic.train()
    avg_eval_colored = cum_eval_colored / cum_cnt
    avg_eval_zero    = cum_eval_zero    / cum_cnt
    return avg_eval_colored, avg_eval_zero

for update_t in range(max_update_t):
    if update_t == 0 or torch.all(done).item():
        try:
            g = next(train_data_iter)
        except:
            train_data_iter = iter(data_loaders["train"])
            g = next(train_data_iter)

        g.set_n_initializer(dgl.init.zero_initializer)
        ob = env.register(g, num_samples = train_num_samples)
        rollout.insert_ob_and_g(ob, g)


    for step_t in range(max_rollout_t):
        # get action and value prediction
        with torch.no_grad():
            (action, 
            action_log_prob, 
            value_pred, 
            ) = actor_critic.act_and_crit(ob, g)

        # step environments
        ob, reward, done, info = env.step(action)

        # insert to rollout
        rollout.insert_tensors(
            ob, 
            action,
            action_log_prob, 
            value_pred, 
            reward, 
            done
            )
        

        if torch.all(done).item():
            avg_colored = info['colored'].max(dim = 1)[0].mean().cpu()
            avg_zero = info['zero'].max(dim=1)[0].mean().cpu()
            break

    # compute gamma-decayed returns and corresponding advantages
    rollout.compute_rets_and_advantages(gamma)

    # update actor critic model with ppo
    actor_loss, critic_loss, entropy_loss = framework.update(rollout)


    # log stats
    if (update_t + 1) % log_freq == 0:
        print("update_t: {:05d}".format(update_t + 1))
        print("train stats...")
        print(
            "colored: {:.4f}, "
            "zero: {:.4f}, "
            "actor_loss: {:.4f}, "
            "critic_loss: {:.4f}, "
            "entropy: {:.4f}".format(
                avg_colored,
                avg_zero,
                actor_loss.item(),
                critic_loss.item(),
                entropy_loss.item()
                )
            )
        
        if (update_t + 1) % vali_freq == 0:
            colored, zero = evaluate("vali", actor_critic)
            print("vali stats...")
            print("colored: {:.4f}".format(colored.item()))
            print("zero: {:.4f}".format(zero.item()))


