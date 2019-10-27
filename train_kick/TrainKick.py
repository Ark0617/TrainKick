import gym
from gym import spaces
import numpy as np
import time

from train_kick.env.Robot import NeoRobot
from train_kick.env.Logger import Logger
from train_kick.env.GrandEnvs import GrandEnvs
from train_goalie.env.GrandEnvs import NotFoundPlayerException
from train_kick.env.TCPClient import AgentConnection


class TrainKick(gym.Env):

    logger = Logger.getLogger("TrainKick")
    metadata = {'render.modes': ['human']}
    trainTypes = {'standup', 'kick'}
    STEPWAIT = 1
    RESETSTEP = 30

    def __init__(self, env_id=1, serverIp="192.168.0.16", serverPort=3100, monitorPort=3200, team="sydney1",
                 playerNumber=0, locationX=10, locationY=10, sleep_time=1, max_episode_steps=200, trainType='kick'):
        super(TrainKick, self).__init__()
        self.env_id = env_id
        self.sum_rewards = 0
        self.sum_action_rewards = 0
        self.locationX = 2 * locationX
        self.locationY = 2 * locationY
        self.serverIp = serverIp
        self.serverPort = serverPort
        self.monitorPort = monitorPort
        self.team = team
        self.playerNumber = playerNumber
        self.max_episode_steps = max_episode_steps
        self.trainType = trainType
        self.rewards_list = []
        self.robot = NeoRobot(self.team, self.playerNumber, self.env_id, self.locationX, self.locationY)
        self.init_ball_location = self.robot.set_ball_nearyby()
        time.sleep(sleep_time)
        self.robot.joinGame(self.serverIp, self.serverPort, self.monitorPort)
        action_dim = 20
        high = np.ones([action_dim])
        self.action_space = spaces.Box(-high, high, dtype=np.float64)
        high = 10 * np.ones(len(self.robot.getAllStates()))
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)
        self.frame = 0
        self.done = False
        self.after_action = time.time()
        self.next_time = time.time() + 0.02 * TrainKick.STEPWAIT
        self.twolegs = 1
        self.epstart = False
        self.isFalled = False
        self.isStand = False
        self.standLast = 0
        self.monitorLastTime = -1

    def step(self, actionParameters):
        self.frame += 1
        if self.frame >= self.max_episode_steps:
            self.episode_over()
        reward = 0
        if self.frame > self.RESETSTEP:
            self.robot.action(actionParameters)
            reward = self.__reward()
        elif self.frame == self.RESETSTEP - 12:
            self.__sendResetCommand()
        else:
            states = self.robot.getAllStates(1)
            self.__resetAction(states)
        self.wait()
        states = self.robot.getAllStates(1)
        if np.any(np.isnan(states)):
            self.logger.error("There is a Nan in states. " + str(states))
        return states, reward, self.done, {}

    def episode_over(self):
        self.done = True

    def __getRobotHeight(self):
        if self.monitorLastTime != self.robot.grandEnvs.lastReadTime and self.robot.grandEnvs.lastReadTime > 0:
            self.monitorLastTime = self.robot.grandEnvs.lastReadTime
            if self.robot.team == 'sydney1':
                team = 0
            else:
                team = 1
            try:
                location = self.robot.grandEnvs.getPlayerLocation(team, self.playerNumber)
            except NotFoundPlayerException as e:
                self.logger.exception("NotFoundPlayerException" + repr(e))
                self.__rebootRobot()
                return None
            if location is not None:
                return location
            else:
                return None
        else:
            return None

    def __rebootRobot(self):
        self.robot.close()
        self.episode_over()

        time.sleep(10)
        self.robot = NeoRobot(self.team, self.playerNumber, self.env_id)
        self.robot.joinGame(self.serverIp, self.serverPort, self.monitorPort)

    def __reward(self):
        if self.trainType not in self.trainTypes:
            self.logger.error("train type error!!!")
            exit(1)

        live_reward = self.__liveReward()
        if self.trainType == 'kick':
            action_reward = self.__rewardKick()
        elif self.trainType == 'standup':
            action_reward = self.__rewardStandUp()

        reward = live_reward + action_reward
        self.sum_action_rewards += action_reward
        self.sum_rewards += reward
        self.rewards_list.append(reward)
        return reward

    def __liveReward(self):
        if self.sum_action_rewards > 0:
            reward = 0.01
        else:
            reward = 0
        return reward

    def __rewardKick(self):
        if self.robot.grandEnvs.getBallLocation() - self.init_ball_location > 0.1:
            reward = self.robot.grandEnvs.getBallLocation() - self.init_ball_location
        else:
            reward = -1
        return reward

    def __rewaredStandUp(self):
        height = self.__getRobotHeight()
        if height <= 0:
            return 0

        if height < 0.2 and self.epstart is False:
            self.epstart = True
        if self.epstart is False:
            return 0

        reward = 0
        if height < 0.2 and self.isStand is False:
            self.lieLast += 1
        elif height > 0.3:
            self.isStand = True
        if height > 0.45:
            reward += 1
        return reward

    def reset(self):
        self.frame = 0
        self.epstart = False
        self.done = False
        self.standLast = 0
        self.lieLast = 0
        self.sum_rewards = 0
        self.sum_action_rewards = 0
        self.rewards_list = []
        self.isStand = False
        self.isFalled = False
        self.monitorLastTime = -1
        self.lastWalkTime = 0
        self.lastWalkPlace = None
        self.twolegs = 1
        self.init_ball_location = self.robot.set_ball_nearby()
        if self.RESETSTEP < 2:
            self.__doReset()
            self.__sendResetCommand()
            self.wait(2)

        states = self.robot.getAllStates(1)
        if np.any(np.isnan(states)):
            self.logger.error("There is a Nan in states. "+ str(states))
        return states

    def __doReset(self):
        states = self.robot.getAllStates()
        count = 0
        while self.__checkResetAction(states) and count < 100:
            self.__resetAction(states)
            self.wait()
            states = self.robot.getAllStates()
            count += 1
        actionParameters = np.zeros(20)
        self.robot.action(actionParameters)
        self.wait()
        self.logger.debug("reset count:" + str(count))

    def __sendResetCommand(self):
        self.robot.con.sendMessage("(beam " + str(self.locationX) + " " + str(self.locationY) + " 0.0)")
        self.init_ball_location = self.robot.set_ball_nearby()

    def __checkResetAction(self, states):
        def __checkResetAction(self, states):
            for i in range(20):
                pstate = states[i + 8]
                if (i == 0 or i == 16) and (pstate < -95 or pstate > -85):
                    # print(str(i) + "pstate" + str(pstate))
                    return True
                elif (i == 6 or i == 12) and (pstate < -1.84 or pstate > 4.84):
                    # print(str(i) + "pstate" + str(pstate))
                    return True
                elif i != 6 and i != 12 and i != 0 and i != 16 and (pstate > 3 or pstate < -3):
                    # print(str(i) + "pstate" + str(pstate))
                    return True
            return False

    def __resetAction(self, states):
        actionParameters = np.zeros(20)
        # self.logger.info("reset body:" + str(states))
        for i in range(20):
            pstate = states[i+8]
            if i == 0 or i == 16:
                pstate += 90
            if i == 6 or i == 12:
                pstate -= 1.84

            if pstate != 0:
                if pstate >= 30:
                    action = -1
                elif pstate <= -30:
                    action = 1
                elif pstate >= 10 and pstate < 30:
                    action = -1/3
                elif pstate > 0 and pstate < 10:
                    action = -pstate/70
                elif pstate < -10 and pstate > -30:
                    action = 1/3
                elif pstate < 0 and pstate > -10:
                    action = -pstate/70
                else:
                    action = 0

                actionParameters[i] = action
        self.robot.action(actionParameters)

    def wait(self, loop=1):
        for i in range(loop):
            self.after_action = time.time()
            self.next_time += 0.02 * TrainKick.STEPWAIT
            sleeptime = self.next_time - self.after_action
            if (sleeptime > 0):
                if (sleeptime > 0.01):
                    sleepTime = sleeptime - 0.01
                else:
                    sleepTime = 0.001
            else:
                self.next_time = self.after_action + 0.02 * TrainKick.STEPWAIT
                sleepTime = 0.02 * TrainKick.STEPWAIT - 0.005
            self.logger.debug("id:" + str(self.env_id) + " " + str(time.time()) + " next time:" + str(
                self.next_time) + " sleep time:" + str(sleepTime))
            time.sleep(sleepTime)

    def render(self, mode='human', close=False):
        pass