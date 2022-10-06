# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from env import MyCobotEnv
from stable_baselines3 import PPO

# policy_path = "./cnn_policy/jetbot_policy.zip"

my_env = MyCobotEnv(headless=False)
# model = PPO.load(policy_path)

for _ in range(20):
    obs = my_env.reset()
    done = False
    while not done:
        actions = my_env.action_space.sample()
        obs, reward, done, info = my_env.step(actions)
        my_env.render()

my_env.close()
