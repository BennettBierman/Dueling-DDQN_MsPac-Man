import argparse
import math
import logging
import timeit
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import utils as utils

logging.basicConfig(format="[%(levelname)s:%(asctime)s] " "%(message)s", level=logging.INFO)


# Task 2: Create the Model
# FIFO Buffer used to store transitions (state, action, reward, done, next_state)
class ReplayMemory(object):
    def __init__(self, capacity, batch_obs_space):
        self.bz, C, W, H = batch_obs_space
        self.cap, self.i, self.size = capacity*self.bz, 0, 0
        self.state_mem = np.zeros((self.cap, C, W, H))
        self.action_mem = np.zeros(self.cap)
        self.reward_mem = np.zeros(self.cap)
        self.done_mem = np.zeros(self.cap)
        self.next_state_mem = np.zeros((self.cap, C, W, H))

    def push(self, p_states, p_actions, p_rewards, p_dones, p_next_states):
        if self.i + self.bz > self.cap:
            self.i = 0
            print('New Re-write')

        self.state_mem[self.i:self.i+self.bz, :, :, :] = p_states
        self.next_state_mem[self.i:self.i+self.bz, :, :, :] = p_next_states
        self.action_mem[self.i:self.i+self.bz] = p_actions
        self.reward_mem[self.i:self.i + self.bz] = p_rewards
        self.done_mem[self.i:self.i + self.bz] = p_dones

        self.i += self.bz

        if self.size < self.cap:
            self.size += self.bz

    def sample(self, batch_size):
        rows = np.random.permutation(self.size)[:batch_size]
        s_states = torch.tensor(self.state_mem[rows], dtype=torch.float32)
        s_actions = torch.tensor(self.action_mem[rows], dtype=torch.int64).unsqueeze(-1)
        s_rewards = torch.tensor(self.reward_mem[rows], dtype=torch.int64).unsqueeze(-1)
        s_dones = torch.tensor(self.done_mem[rows], dtype=torch.int64).unsqueeze(-1)
        s_next_states = torch.tensor(self.next_state_mem[rows], dtype=torch.float32)
        return s_states, s_actions, s_rewards, s_dones, s_next_states

# Double-Dueling DQN Network
class Network(nn.Module):
    def __init__(self, n_actions, args):
        super().__init__()

        self.n_actions = n_actions

        self.conv_net = nn.Sequential(
            nn.Conv2d(args.frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, args.hidden_dim),
            nn.ReLU()
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(args.hidden_dim, self.n_actions)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, states):
        Z = self.conv_net(states)
        vals = self.value_layer(Z)
        advs = self.advantage_layer(Z)
        return vals + (advs - advs.mean(dim=1, keepdim=True))

    # Epsilon Greedy Policy
    def action_batch(self, states, epsilon):
        q_vaules = self(states)
        max_action_indicies = torch.argmax(q_vaules, dim=1)
        policy_actions = max_action_indicies.detach()
        epsilon_mask = (torch.rand(policy_actions.shape) <= epsilon)
        rand_actions = torch.randint(0, self.n_actions - 1, policy_actions.shape)
        policy_actions[epsilon_mask] = rand_actions[epsilon_mask]
        return policy_actions.numpy()

    def compute_loss(self, transitions, target_net):
        states, actions, rewards, dones, next_states = transitions

        # Compute Targets with Double DQN Architecture
        with torch.no_grad():
            online_qvalues = self(next_states)
            max_online_indicies = online_qvalues.argmax(dim=1, keepdim=True)
            target_qvalues = target_net(next_states)
            target_values = target_qvalues.gather(1, max_online_indicies)
            targets = rewards + args.discounting * (1 - dones) * target_values

        # Compute Loss
        q_values = self(states)
        action_q_values = q_values.gather(1, actions)
        return F.huber_loss(action_q_values, targets)

    # Compute policy's action given one observation
    def get_action(self, state):
        q_values = self(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return q_values.argmax(-1).squeeze().item()


# Pre-fill buffer prior to performing gradient-descent
def prep_buffer(buffer, env, rounds):
    states, infos = env.reset()

    for _ in range(rounds):
        actions = env.action_space.sample()
        observations, rewards, terms, truncs, infos = env.step(actions)
        dones = terms + truncs
        buffer.push(states, actions, rewards, dones.astype(int), observations)
        states = observations

    return buffer


# Task 1: Create Learning Algorithm - DQN Training
def pg_step(env, online_net, target_net, buffer, optimizer, scheduler, epsilon, time, bsz=4):

    states, infos = env.reset()
    reward_sum = 0
    total_loss = 0

    # Collect transitions
    for t in range(time):
        actions = online_net.action_batch(torch.as_tensor(states, dtype=torch.float32), epsilon)
        observations, rewards, terms, truncs, infos = env.step(actions)
        dones = terms + truncs
        reward_sum += np.sum(rewards*(1-dones))
        buffer.push(states, actions, rewards, dones.astype(int), observations)
        states = observations

        # Start Gradient Step
        transitions = buffer.sample(bsz)
        loss = online_net.compute_loss(transitions, target_net)
        total_loss += loss

        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online_net.parameters(), args.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

    stats = {"mean_return": reward_sum / (time * bsz), "pg_loss": total_loss/ (time * bsz), "epsilon":epsilon}
    return stats, env, online_net, buffer


def train(args):
    T = args.unroll_length  # time
    B = args.batch_size  # Batches - number of parallel environments
    args.device = torch.device("cpu")

    repeat_env = FrameStack(
        AtariPreprocessing(gym.make(args.env), frame_skip=1, scale_obs=True),
        num_stack=args.frame_stack
    )

    batch_env = gym.vector.SyncVectorEnv([
        lambda: repeat_env for _ in range(B)
    ])

    naction = repeat_env.action_space.n
    args.start_nlives = repeat_env.ale.lives()

    online_net = Network(naction, args)
    target_net = Network(naction, args)
    target_net.load_state_dict(online_net.state_dict())

    def lr_lambda(epoch):  # multiplies learning rate by value returned; can be used to decay lr
        return 1

    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    buffer = ReplayMemory(args.buffer_size, batch_env.observation_space.shape)
    buffer = prep_buffer(buffer, batch_env, args.min_buffer)

    def checkpoint():
        if args.save_path is None:
            return

        logging.info("Saving checkpoint to {}".format(args.save_path))
        torch.save({"model_state_dict": online_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": args}, args.save_path)

    timer = timeit.default_timer
    last_checkpoint_time = timer()
    frame = 0
    target_thresh = args.target_updates

    while frame < args.total_frames:
        start_time = timer()
        start_frame = frame

        epsilon = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1. * frame / args.eps_decay)

        # Run Alg
        stats, batch_env, online_net, buffer = pg_step(batch_env, online_net, target_net, buffer,
                                                  optimizer, scheduler, epsilon, T, bsz=B)

        frame += T * B  # here steps means number of observations

        # Update Target Network
        if frame > target_thresh:
            print('targets updated')
            target_net.load_state_dict(online_net.state_dict())
            target_thresh += args.target_updates

        if timer() - last_checkpoint_time > args.min_to_save * 60:
            checkpoint()
            last_checkpoint_time = timer()

        sps = (frame - start_frame) / (timer() - start_time)
        logging.info("Frame {:d} @ {:.1f} FPS: pg_loss {:.3f} | mean_ret {:.3f} | epsilon {:.3f}".format(
            frame, sps, stats['pg_loss'], stats["mean_return"], stats["epsilon"]))

        # NEED TO FIX
        if frame > 0 and frame % (args.eval_every * T * B) == 0:
            utils.validate(online_net, render=args.render, frame_stack=args.frame_stack)
            online_net.train()


# DO NOT TOUCH
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="ALE/MsPacman-v5", help="gym environment")
parser.add_argument("--mode", default="train", choices=["train", "valid"], help="training or validation mode")
parser.add_argument("--total_frames", default=1000000, type=int, help="total environment frames to train for")
parser.add_argument("--batch_size", default=16, type=int, help="learner batch size.")
parser.add_argument("--frame_stack", default=4, type=int, help="number of stacked images.")
parser.add_argument("--buffer_size", default=4000, type=int, help="max size of replay memory.")
parser.add_argument("--min_buffer", default=2000, type=int, help="random iterations before learning")
parser.add_argument("--eps_start", default=0.99, type=float, help="greedy epsilon starting value")
parser.add_argument("--eps_end", default=0.02, type=float, help="greedy epsilon ending value")
parser.add_argument("--eps_decay", default=800000, type=float, help="steps it takes for eps to decay")
parser.add_argument("--target_updates", default=20000, type=int, help="how often target_net is updated")
parser.add_argument("--unroll_length", default=100, type=int, help="unroll length (time dimension)")
parser.add_argument("--hidden_dim", default=512, type=int, help="policy net hidden dim")
parser.add_argument("--discounting", default=0.985, type=float, help="discounting factor")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
parser.add_argument("--grad_norm_clipping", default=10.0, type=float, help="Global gradient norm clip.")
parser.add_argument("--save_path", type=str, default=None, help="save model here")
parser.add_argument("--load_path", type=str, default=None, help="load model from here")
parser.add_argument("--min_to_save", default=5, type=int, help="save every this many minutes")
parser.add_argument("--eval_every", default=50, type=int, help="eval every this many updates")
parser.add_argument("--render", action="store_true", help="render game-play at validation time")

if __name__ == "__main__":
    torch.manual_seed(59006)
    np.random.seed(59006)
    args = parser.parse_args()
    logging.info(args)

    if args.mode == "train":
        train(args)

    else:
        assert args.load_path is not None

        checkpoint = torch.load(args.load_path)
        saved_args = checkpoint["args"]

        env = FrameStack(
            AtariPreprocessing(gym.make(args.env), frame_skip=1, scale_obs=True),
            num_stack=args.frame_stack
        )

        naction = env.action_space.n
        saved_args.start_nlives = env.ale.lives()

        del env

        model = Network(naction, saved_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model
        args = saved_args

        utils.validate(model, args)
