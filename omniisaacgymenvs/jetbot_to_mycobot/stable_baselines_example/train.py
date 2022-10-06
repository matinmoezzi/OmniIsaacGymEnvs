# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from xmlrpc.client import boolean
from env import MyCobotEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import argparse

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
parser.add_argument("--headless", default=False, type=boolean, help="Whether not to monitor the training process visually")
args, unknown = parser.parse_known_args()

log_dir = "./cnn_policy"
# set headless to false to visualize training
my_env = MyCobotEnv(headless=False)


policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[16, dict(pi=[64, 32], vf=[64, 32])])
total_timesteps = 500000

if args.test is True:
    total_timesteps = 10000

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="mycobot_policy_checkpoint")

model = PPO(
    CnnPolicy,
    my_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=10000,
    batch_size=1000,
    learning_rate=0.00025,
    gamma=0.9995,
    device="cuda",
    ent_coef=0,
    vf_coef=0.5,
    max_grad_norm=10,
    tensorboard_log=log_dir,
)

# model = PPO(
#     MlpPolicy,
#     my_env,
#     policy_kwargs=policy_kwargs,
#     verbose=1,
#     n_steps=2560,
#     batch_size=64,
#     learning_rate=0.000125,
#     gamma=0.9,
#     ent_coef=7.5e-08,
#     clip_range=0.3,
#     n_epochs=5,
#     gae_lambda=1.0,
#     max_grad_norm=0.9,
#     vf_coef=0.95,
#     device="cuda",
#     tensorboard_log=log_dir,
# )

model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

model.save(log_dir + "/mycobot_policy")

my_env.close()

# If you try to use CnnPolicy as the MlpPolicy, you need to follow the parameters setting provided in Isaac Sim - 2022.1.0
# Paste here