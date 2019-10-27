from gym.envs.registration import register
from train_kick.TrainKick import TrainKick
from train_kick.env.Robot import NeoRobot
from train_kick.env.Logger import Logger
from train_kick.env.TCPClient import AgentConnection


__all__ = ["Logger", "NeoRobot", "AgentConnection", "TrainKick"]

register(
      id='TrainKick-v0',
      entry_point='train_kick:TrainKick',
      max_episode_steps=1000
)
