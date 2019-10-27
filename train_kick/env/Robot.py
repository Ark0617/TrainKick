#!/usr/bin/python3

# print(os.getcwd())
import time
import re
from enum import Enum, unique

from train_kick.env.Logger import Logger
from train_kick.env.TCPClient import AgentConnection
from train_kick.env.GrandEnvs import GrandEnvs
from train_kick.env.GrandEnvs import NotFoundPlayerException

"""
This is the robot basic model for all kinds of models in RoboCup
Other specific model would extend this model to avoid redundant codes
"""


@unique
class NeoTypes(Enum):
    NAO = "(scene rsg/agent/nao/nao.rsg)"
    NAO1 = "(scene rsg/agent/nao/nao_hetero.rsg 1)"
    NAO2 = "(scene rsg/agent/nao/nao_hetero.rsg 2)"
    NAO3 = "(scene rsg/agent/nao/nao_hetero.rsg 3)"
    NAOTOE = "(scene rsg/agent/nao/nao_hetero.rsg 4)"


PERCEPTOR = Enum(
    'PERCEPTOR',
    ('hj1', 'hj2', 'laj1', 'laj2', 'laj3', 'laj4', 'llj1', 'llj2', 'llj3',
     'llj4', 'llj5', 'llj6', 'rlj1', 'rlj2', 'rlj3', 'rlj4', 'rlj5', 'rlj6',
     'raj1', 'raj2', 'raj3', 'raj4', 'llj7', 'rlj7'))


class NeoRobot:

    logger = Logger.getLogger("NeoRobot")

    EFFECTOR = [
        'he1', 'he2', 'lae1', 'lae2', 'lae3', 'lae4', 'lle1', 'lle2', 'lle3',
        'lle4', 'lle5', 'lle6', 'rle1', 'rle2', 'rle3', 'rle4', 'rle5',
        'rle6', 'rae1', 'rae2', 'rae3', 'rae4', 'lle7', 'rle7'
    ]
    """
    (max joint speed 7:035deg=cycle)
    The range of effectors will be various in different kinds of robots.
    """
    EFFECTOR_RANGE = [7] * 24

    # (GYR (n torso) (rt 0.03 0.00 0.00))
    # (ACC (n torso) (a -0.00 -0.00 0.04))
    gyrPattern = re.compile(
        r'\(GYR\s\(n\storso\)\s\(rt\s(\-?\d+\.?\d+?)\s(\-?\d+\.?\d+?)\s(\-?\d+\.?\d+?)\)\)'
    )
    accPattern = re.compile(
        r'\(ACC\s\(n\storso\)\s\(a\s(\-?\d+\.?\d+?)\s(\-?\d+\.?\d+?)\s(\-?\d+\.?\d+?)\)\)'
    )

    perceptionPattern = re.compile(r'\(HJ\s\(n\s([^()]*)\)\s\(ax\s(.*?)\)\)')

    lfPattern = re.compile(
        r'\(FRP\s\(n\slf\)\s\(c\s(\-?\d+\.?\d+?)\s(\-?\d+\.?\d+?)\s(\-?\d+\.?\d+?)\)\s\(f\s(\-?\d+\.?\d+?)\s(\-?\d+\.?\d+?)\s(\-?\d+\.?\d+?)\)\)'
    )
    rfPattern = re.compile(
        r'\(FRP\s\(n\srf\)\s\(c\s(\-?\d+\.?\d+?)\s(\-?\d+\.?\d+?)\s(\-?\d+\.?\d+?)\)\s\(f\s(\-?\d+\.?\d+?)\s(\-?\d+\.?\d+?)\s(\-?\d+\.?\d+?)\)\)'
    )

    def __init__(self, team, playerNumber, env_id=1, locationX=1, locationY=1, robotType=NeoTypes.NAO):
        self.logger.info("init robot " + team + ":" + str(playerNumber))

        # Instance Attribute
        self.robotType = robotType
        self.team = team
        self.name = str(env_id) + ":" + team + ":" + str(playerNumber)
        self.playerNumber = playerNumber
        self.locationX = locationX
        self.locationY = locationY
        self.grandEnvs = GrandEnvs()
        """
        (GYR (n torso) (rt 0.00 0.00 0.00)) GyroRatePerceptor
        (ACC (n torso) (a -0.00 -0.00 9.81)) Accelerometer
        """
        self.GYR = [.0, .0, .0]
        self.ACC = [.0, .0, .0]
        """
        (HJ (n hj1) (ax -0.00))
        (HJ (n hj2) (ax -0.00))
        """
        self.neckYaw = .0
        self.neckPitch = .0

        self.speedNeckYaw = .0
        self.speedNeckPitch = .0
        '''
        (HJ (n laj1) (ax 0.00)) Left Shoulder Pitch
        (HJ (n laj2) (ax 0.00)) Left Shoulder Yaw
        (HJ (n laj3) (ax 0.00)) Left Shoulder Roll
        (HJ (n laj4) (ax 0.00)) Left Arm Yaw
        '''
        self.leftShoulderPitch = .0
        self.leftShoulderYaw = .0
        self.leftShoulderRoll = .0
        self.leftArmYaw = .0

        self.speedLeftShoulderPitch = .0
        self.speedLeftShoulderYaw = .0
        self.speedLeftShoulderRoll = .0
        self.speedLeftArmYaw = .0
        """
        (HJ (n llj1) (ax 0.00)) Left Hip YawPitch
        (HJ (n llj2) (ax 0.00)) Left Hip Yaw
        (HJ (n llj3) (ax 1.84)) Left Hip Roll
        (HJ (n llj4) (ax 0.00)) Left Knee Pitch
        (HJ (n llj5) (ax 0.00)) Left Foot Yaw
        (HJ (n llj6) (ax -0.00)) Left Foot Roll
        """
        self.leftHipYawPitch = .0
        self.leftHipRoll = .0
        self.leftHipPitch = .0
        self.leftKneePitch = .0
        self.leftFootPitch = .0
        self.leftFootRoll = .0

        self.speedLeftHipYawPitch = .0
        self.speedLeftHipRoll = .0
        self.speedLeftHipPitch = .0
        self.speedLeftKneePitch = .0
        self.speedLeftFootPitch = .0
        self.speedLeftFootRoll = .0
        """
        (HJ (n rlj1) (ax 0.00)) Right Hip YawPitch
        (HJ (n rlj2) (ax 0.00)) Right Hip Roll
        (HJ (n rlj3) (ax 1.84)) Right Hip Pitch
        (HJ (n rlj4) (ax 0.00)) Right Knee Pitch
        (HJ (n rlj5) (ax 0.00)) Right Foot Pitch
        (HJ (n rlj6) (ax 0.00)) Right Foot Roll
        """
        self.rightHipYawPitch = .0
        self.rightHipRoll = .0
        self.rightHipPitch = .0
        self.rightKneePitch = .0
        self.rightFootPitch = .0
        self.rightFootRoll = .0

        self.speedRightHipYawPitch = .0
        self.speedRightHipRoll = .0
        self.speedRightHipPitch = .0
        self.speedRightKneePitch = .0
        self.speedRightFootPitch = .0
        self.speedRightFootRoll = .0
        """
        (HJ (n raj1) (ax 0.00)) Right Shoulder Pitch
        (HJ (n raj2) (ax 0.00)) Right Shoulder Yaw
        (HJ (n raj3) (ax 0.00)) Right Shoulder Roll
        (HJ (n raj4) (ax -0.00)) Right Arm Yaw
        """
        self.rightShoulderPitch = .0
        self.rightShoulderYaw = .0
        self.rightShoulderRoll = .0
        self.rightArmYaw = .0

        self.speedRightShoulderPitch = .0
        self.speedRightShoulderYaw = .0
        self.speedRightShoulderRoll = .0
        self.speedRightArmYaw = .0
        """
        Only for Nao Toe robot
        """
        self.rightToe = None
        self.leftToe = None

        self.speedRightToe = .0
        self.speedLeftToe = .0
        """
        (FRP (n rf) (c 0.00 -0.01 -0.01) (f -0.00 -0.00 22.60))
        (FRP (n lf) (c -0.00 -0.01 -0.01) (f 0.00 -0.00 22.60))
        """
        self.rightForceResistance = [.0, .0, .0, .0, .0, .0]
        self.leftForceResistance = [.0, .0, .0, .0, .0, .0]

    def joinGame(self, ip="192.168.0.16", port=3100, mport=3200):
        self.con = AgentConnection(id, self.name, self)
        self.con.connectServers(ip, port, mport)
        self.con.start()
        """
        Init robot in the game with game commands.
        """

        self.con.sendMessage(self.robotType.value)

        time.sleep(1)
        self.con.sendMessage("(init (unum " + str(self.playerNumber) +
                             ")(teamname " + self.team + "))")
        time.sleep(1)
        self.con.sendMessage("(beam " + str(self.locationX) + " " + str(self.locationY) + " 0.0)")
        time.sleep(1)

    def close(self):
        self.con.close()

    def hasFalled(self):
        if ((abs(self.ACC[0]) > 7
                or abs(self.ACC[1] > 7)) and self.ACC[2] < 3):
            return True
        else:
            return False

    def set_ball_nearby(self):
        ball_loc = [self.locationX + 0.1, self.locationY + 0.1, 0]
        self.con.sendMessage("(ball (pos " + str(ball_loc[0])+" "+str(ball_loc[1])+" "+str(ball_loc[2])+")(vel 0 0 0))")
        return ball_loc

    def update(self, message):

        try:
            match1 = NeoRobot.gyrPattern.search(message)
            match2 = NeoRobot.accPattern.search(message)

            self.GYR[0] = float(match1.group(1))
            self.GYR[1] = float(match1.group(2))
            self.GYR[2] = float(match1.group(3))
            self.ACC[0] = float(match2.group(1))
            self.ACC[1] = float(match2.group(2))
            self.ACC[2] = float(match2.group(3))

            it = NeoRobot.perceptionPattern.finditer(message)
            for match in it:
                if (match.group(1) == PERCEPTOR.hj1.name):
                    self.speedNeckYaw = float(match.group(2)) - self.neckYaw
                    self.neckYaw = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.hj2.name):
                    self.speedNeckPitch = float(match.group(2)) - self.neckPitch
                    self.neckPitch = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.laj1.name):
                    self.speedLeftShoulderPitch = float(match.group(2)) - self.leftShoulderPitch
                    self.leftShoulderPitch = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.laj2.name):
                    self.speedLeftShoulderYaw = float(match.group(2)) - self.leftShoulderYaw
                    self.leftShoulderYaw = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.laj3.name):
                    self.speedLeftShoulderRoll = float(match.group(2)) - self.leftShoulderRoll
                    self.leftShoulderRoll = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.laj4.name):
                    self.speedLeftArmYaw = float(match.group(2)) - self.leftArmYaw
                    self.leftArmYaw = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.llj1.name):
                    self.speedLeftHipYawPitch = float(match.group(2)) - self.leftHipYawPitch
                    self.leftHipYawPitch = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.llj2.name):
                    self.speedLeftHipRoll = float(match.group(2)) - self.leftHipRoll
                    self.leftHipRoll = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.llj3.name):
                    self.speedLeftHipPitch = float(match.group(2)) - self.leftHipPitch
                    self.leftHipPitch = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.llj4.name):
                    self.speedLeftKneePitch = float(match.group(2)) - self.leftKneePitch
                    self.leftKneePitch = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.llj5.name):
                    self.speedLeftFootPitch = float(match.group(2)) - self.leftFootPitch
                    self.leftFootPitch = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.llj6.name):
                    self.speedLeftFootRoll = float(match.group(2)) - self.leftFootRoll
                    self.leftFootRoll = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.llj7.name):
                    self.speedLeftToe = float(match.group(2)) - self.leftToe
                    self.leftToe = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.rlj1.name):
                    self.speedRightHipYawPitch = float(match.group(2)) - self.rightHipYawPitch
                    self.rightHipYawPitch = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.rlj2.name):
                    self.speedRightHipRoll = float(match.group(2)) - self.rightHipRoll
                    self.rightHipRoll = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.rlj3.name):
                    self.speedRightHipPitch = float(match.group(2)) - self.rightHipPitch
                    self.rightHipPitch = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.rlj4.name):
                    self.speedRightKneePitch = float(match.group(2)) - self.rightKneePitch
                    self.rightKneePitch = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.rlj5.name):
                    self.speedRightFootPitch = float(match.group(2)) - self.rightFootPitch
                    self.rightFootPitch = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.rlj6.name):
                    self.speedRightFootRoll = float(match.group(2)) - self.rightFootRoll
                    self.rightFootRoll = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.rlj7.name):
                    self.speedRightToe = float(match.group(2)) - self.rightToe
                    self.rightToe = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.raj1.name):
                    self.speedRightShoulderPitch = float(match.group(2)) - self.rightShoulderPitch
                    self.rightShoulderPitch = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.raj2.name):
                    self.speedRightShoulderYaw = float(match.group(2)) - self.rightShoulderYaw
                    self.rightShoulderYaw = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.raj3.name):
                    self.speedRightShoulderRoll = float(match.group(2)) - self.rightShoulderRoll
                    self.rightShoulderRoll = float(match.group(2))
                elif (match.group(1) == PERCEPTOR.raj4.name):
                    self.speedRightArmYaw = float(match.group(2)) - self.rightArmYaw
                    self.rightArmYaw = float(match.group(2))
            """
            Both of parameters(lf, rf) are not awalys in the message.
            """
            lfMatch = NeoRobot.lfPattern.search(message)
            rfMatch = NeoRobot.rfPattern.search(message)

            if (lfMatch is not None):
                self.leftForceResistance[0] = float(lfMatch.group(1))
                self.leftForceResistance[1] = float(lfMatch.group(2))
                self.leftForceResistance[2] = float(lfMatch.group(3))
                self.leftForceResistance[3] = float(lfMatch.group(4))
                self.leftForceResistance[4] = float(lfMatch.group(5))
                self.leftForceResistance[5] = float(lfMatch.group(6))
            else:
                self.leftForceResistance = [.0, .0, .0, .0, .0, .0]

            if (rfMatch is not None):
                self.rightForceResistance[0] = float(rfMatch.group(1))
                self.rightForceResistance[1] = float(rfMatch.group(2))
                self.rightForceResistance[2] = float(rfMatch.group(3))
                self.rightForceResistance[3] = float(rfMatch.group(4))
                self.rightForceResistance[4] = float(rfMatch.group(5))
                self.rightForceResistance[5] = float(rfMatch.group(6))
            else:
                self.rightForceResistance = [.0, .0, .0, .0, .0, .0]

        except AttributeError as e:
            self.logger.exception("socket.error" + repr(e))
            self.logger.error("message:" + message)

    def actionToMessage(self, effectors):
        """
        The number of effectors must be 24, or the order of arrays will be wrong
        """
        message = ""
        for i in range(24):
            rate = self.EFFECTOR_RANGE[i] * effectors[i]
            # if (rate > 0.1 or rate < -0.1):
            message += '(%s %.2f)' % (self.EFFECTOR[i], rate)

        return message

    def action(self, effectors):
        """
        Change effectors to right foramt with 24 parameters adding head and toe
        """
        effectors2 = [.0, .0]
        effectors2.extend(effectors)
        effectors2.extend([.0, .0])
        message = self.actionToMessage(effectors2)
        self.con.sendMessage(message)

    def getAllStates(self, reset=0):
        """
        GYR 3 + ACC 3 + 22 States + Toe 2 + rl 6 + lf 6 = 42
        Speed 24 + Reset sate 1 + location 3 = 70
        """
        states = [
            self.GYR[0], self.GYR[1], self.GYR[2], self.ACC[0], self.ACC[1],
            self.ACC[2]
        ]
        states += [self.neckYaw, self.neckPitch]
        states += [
            self.leftShoulderPitch, self.leftShoulderYaw,
            self.leftShoulderRoll, self.leftArmYaw
        ]
        states += [
            self.leftHipYawPitch, self.leftHipRoll, self.leftHipPitch,
            self.leftKneePitch, self.leftFootPitch, self.leftFootRoll
        ]
        states += [
            self.rightHipYawPitch, self.rightHipRoll, self.rightHipPitch,
            self.rightKneePitch, self.rightFootPitch, self.rightFootRoll
        ]
        states += [
            self.rightShoulderPitch, self.rightShoulderYaw,
            self.rightShoulderRoll, self.rightArmYaw
        ]

        if (self.rightToe is not None and self.leftToe is not None):
            states += [self.rightToe] + [self.leftToe]
        else:
            states += [0] + [0]
        """
        Append rf and lf into states
        """
        states += [
            self.rightForceResistance[0], self.rightForceResistance[1],
            self.rightForceResistance[2], self.rightForceResistance[3],
            self.rightForceResistance[4], self.rightForceResistance[5]
        ]
        states += [
            self.leftForceResistance[0], self.leftForceResistance[1],
            self.leftForceResistance[2], self.leftForceResistance[3],
            self.leftForceResistance[4], self.leftForceResistance[5]
        ]

        """
        Speed
        """
        states += [self.speedNeckYaw, self.speedNeckPitch]
        states += [
            self.speedLeftShoulderPitch, self.speedLeftShoulderYaw,
            self.speedLeftShoulderRoll, self.speedLeftArmYaw
        ]
        states += [
            self.speedLeftHipYawPitch, self.speedLeftHipRoll, self.speedLeftHipPitch,
            self.speedLeftKneePitch, self.speedLeftFootPitch, self.speedLeftFootRoll
        ]
        states += [
            self.speedRightHipYawPitch, self.speedRightHipRoll, self.speedRightHipPitch,
            self.speedRightKneePitch, self.speedRightFootPitch, self.speedRightFootRoll
        ]
        states += [
            self.speedRightShoulderPitch, self.speedRightShoulderYaw,
            self.speedRightShoulderRoll, self.speedRightArmYaw
        ]

        if (self.rightToe is not None and self.leftToe is not None):
            states += [self.speedRightToe] + [self.speedLeftToe]
        else:
            states += [0] + [0]

        """
        reset
        """
        states += [reset]
        """
        locationX, Y, Z
        """
        location = self.__getRobotLocation()
        states += [location[0], location[1], location[2]]

        return states

    def getRobotPitch(self):
        """
        This method may get old data, but it's better than None
        """
        if(self.grandEnvs.lastReadTime > 0):
            if(self.team == "sydney1"):
                team = 0
            else:
                team = 1
            try:
                pitch = self.grandEnvs.getPitch(team, self.playerNumber)
                return pitch
            except NotFoundPlayerException as e:
                self.logger.exception("NotFoundPlayerException" + repr(e))

        return -1

    def __getRobotLocation(self):
        """
        This method may get old data, but it's better than None
        """
        if(self.grandEnvs.lastReadTime > 0):
            if(self.team == "sydney1"):
                team = 0
            else:
                team = 1
            try:
                location = self.grandEnvs.getPlayerLocation(team, self.playerNumber)
                return location
            except NotFoundPlayerException as e:
                self.logger.exception("NotFoundPlayerException" + repr(e))

        return [-1000, -1000, -1000]

    def __str__(self):
        return self.name
