import time
import sys
import os
import subprocess
import numpy as np

from stable_baselines import PPO2, TRPO
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

from train_kick import TrainKick
from train_kick.env.Logger import Logger

# IP = "localhost"
# IP = "192.168.0.16"
IP = "127.0.0.1"
port = 3100
mport = 3200
team1 = "sydney1"
team2 = "sydney2"

server_num = 1
num_actors = 1
env_id = "TrainKick-v0"
# loadFile = "SaveModel/model5.pkl"
loadFile = "SaveModel/kicksave3.pkl"
trainType = "kick"

# Create log dir
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
logger = Logger.getLogger("NeoRobot")

if (len(sys.argv) == 3):
    server_num = int(sys.argv[1])
    num_actors = int(sys.argv[2])
    logger.info("server_num:" + str(server_num) + " num_actors:" + str(num_actors))


def startServers():
    cmd = "rm -rf ./logs/*"
    logger.info(cmd)
    subprocess.Popen(cmd, shell=True)

    for j in range(server_num):
        port1_n = port + j
        port2_n = mport + j
        cmd = "simspark --agent-port " + str(port1_n) + " --server-port " + str(port2_n) + " > logs/server" + str(
            j) + ".log 2>&1"
        logger.info(cmd)
        subprocess.Popen(cmd, shell=True)
        time.sleep(2)


monList = []


def get_env(rank, teamname, playerNumber, portj, mportj, sleepTime, max_episode_steps, trainType="kick", seed=0):
    """
    Because the server can accept agent connecting at same time,
    SleepTime makes the init become in order.

    def __init__(self, env_id=1, serverIp="192.168.0.16", serverPort=3100, team="sydney1",
                playerNumber=0, locationX=10, locationY=10, sleep_time=0, max_episode_steps=1000):
    """

    def _get():
        locationX = -playerNumber * 1.5
        locationY = -6 + playerNumber * 1.5
        set_global_seeds(seed + rank)

        env = TrainKick(rank, IP, portj, mportj,
                          teamname, playerNumber,
                          locationX, locationY,
                          sleepTime, max_episode_steps=500, trainType=trainType)

        env.seed(seed + rank)
        logdir = os.path.join(log_dir, str(rank))
        env = Monitor(env, str(logdir), allow_early_resets=True)

        return env

    return _get


"""
Due to the number 5 of each type of robots in each team,
Using two teams could help the number reach to 10.
"""
startServers()
if (server_num > 1):
    time.sleep(120)

envlist = []
for j in range(server_num):
    for i in range(num_actors):
        rank = 100 * j + i + 1
        logdir = os.path.join(log_dir, str(rank)) + ".monitor.csv"
        monList.append(logdir)
        if (i < 5):
            envlist.append(
                get_env(rank,
                        team1,
                        playerNumber=i + 1,
                        portj=j + port,
                        mportj=j + mport,
                        sleepTime=i + 1, max_episode_steps=250, trainType=trainType))
        else:
            envlist.append(
                get_env(rank,
                        team2,
                        playerNumber=i + 1 - 5,
                        portj=j + port,
                        mportj=j + mport,
                        sleepTime=i + 1, max_episode_steps=250, trainType=trainType))

best_mean_reward, n_steps = 0, 0


def callback(_locals, _globals):
    """
    Callback called at each step
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if n_steps % 5 == 0:
        """
        # Evaluate policy training performance
        • timesteps – (Pandas DataFrame) the input data
        • xaxis – (str) the axis for the x and y output (can be X_TIMESTEPS=’timesteps’,
        X_EPISODES=’episodes’ or X_WALLTIME=’walltime_hrs’)
        """
        # for ldir in monList:

        x, y = ts2xy(load_results(log_dir), 'episodes')
        if len(x) > 0:
            lines = server_num * num_actors * -100
            mean_reward = round(np.mean(y[lines:]), 2)

            logger.info(
                "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward - best_mean_reward > 1:
                best_mean_reward = mean_reward
                # Example for saving best model
                savename = 'Model' + str(int(mean_reward)) + '.pkl'
                logger.info("Saving new best model:" + savename)
                _locals['self'].save('SaveModel/' + savename)
    n_steps += 1
    # Returning False will stop training early
    return True


env = SubprocVecEnv(envlist)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=0.51)
# env = VecCheckNan(env, raise_exception=True)
model = PPO2(MlpPolicy, env, verbose=1)  #PPO2.load(loadFile, env, learning_rate=1e-6)
# print(model.get_parameters())
# print(monList)
# Logs will be saved in log_dir/monitor.csv
# env = VecMonitor(env, log_dir, allow_early_resets=True)

# model = TRPO(MlpPolicy, env, verbose=1)

# aenv = envlist[0]
# model = SAC('LnMlpPolicy', aenv, verbose=1)
# model = PPO2(MlpPolicy, env, n_steps=100, nminibatches=5, verbose=1)
# batch = server_num * num_actors
# print(batch)
# model = PPO2(MlpPolicy, env, n_steps=50, learning_rate=0.00005, nminibatches=batch, verbose=1)

for index in range(500):
    model.learn(total_timesteps=int(1e6), log_interval=100, tb_log_name='tb.log', callback=callback)
    model.save('SaveModel/kicksave' + str(index))
    logger.info('SaveModel/kicksave' + str(index))
