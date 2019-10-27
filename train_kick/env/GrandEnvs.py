#!/usr/bin/python3

import copy
import re
import time
import math
import numpy as np

from train_kick.env.Logger import Logger


class GrandEnvs:
    """
    The class dosen't need to be instanced, because only one object is needed.
    """
    logger = Logger.getLogger("GrandEnvs")

    def __init__(self):
        self.playersHeadLocations = {}
        self.playersBodyLocations = {}
        self.ballLocation = [.0, .0, .0]
        self.lastReadTime = -1

    """
    (nd TRF (SLT 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.0417605 1)(nd StaticMesh (setVisible 1) (load models/soccerball.obj) (sSc 0.042 0.042 0.042)(resetMaterials soccerball_rcs-soccerball.png)))
    """
    ballPattern = re.compile(r'\(nd\s([^n]*)\(nd\s([^n]*)soccerball\.obj')
    """
    (1) body location (2) num (3) team (4) type (5) head location
    """
    playerPattern = re.compile(r'\(nd ([^n]*)\(nd [^n]*\(nd [^n]*naobody\d\.obj[^r]*\(resetMaterials matNum(\w*) mat(\w*) matType(\d) naowhite\)\)\)\(nd TRF[^R]* TRF [^R]* ([^n]*)\(nd')

    def updateGrandStates(self, message):
        """
        Only Message whose length is over 10k will be parsed.
        """
        self.logger.debug(message)
        if(message.find("naosoccerfield", 1000, 1500) > 0):
            try:
                ballMatch = self.ballPattern.search(message)
                if (ballMatch is not None):
                    hmsg = self.parseLocation(ballMatch.group(1))
                    self.ballLocation = self.parseHomogenousMatrix(hmsg)

                    it = self.playerPattern.finditer(message)
                    for match in it:
                        hmsg = self.parseLocation(match.group(1))
                        blocation = self.parseHomogenousMatrix(hmsg)
                        hmsg = self.parseLocation(match.group(5))
                        hlocation = self.parseHomogenousMatrix(hmsg)

                        key = ""
                        if(match.group(3).strip() == 'Left' and match.group(2).isdigit()):
                            key = "left_" + str(match.group(2))
                        elif(match.group(3).strip() == 'Right' and match.group(2).isdigit()):
                            key = "right_" + str(match.group(2))
                        if(key != ""):
                            self.playersHeadLocations[key] = hlocation
                            self.playersBodyLocations[key] = blocation

                        self.lastReadTime = time.time()
                # else:
                #    self.logger.error("Can't find ball")
            except AttributeError as e:
                self.logger.exception("Monintor parse error" + repr(e))
                self.logger.error("message:" + message)

    def parseLocation(self, message):
        """
        TRF (SLT 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.0417605 1)
        """
        message = message.strip()
        # self.logger.info(message)
        submessage = message[8:-1].strip()
        return submessage

    def parseHomogenousMatrix(self, matrix):
        slist = str(matrix).split(" ")
        return [float(slist[12]), float(slist[13]), float(slist[14])]

    def getPitch(self, team=0, playerNumber=1):
        """
        return pitch between head and body
        if team zero means leftTeam, while one means rightTeam.
        """
        if(team == 0):
            key = "left_" + str(playerNumber)
        else:
            key = "right_" + str(playerNumber)

        hlocation = self.playersHeadLocations.get(key)
        blocation = self.playersBodyLocations.get(key)
        if(blocation is not None and hlocation is not None):
            length = np.sqrt((hlocation[0]-blocation[0])**2 + (hlocation[1]-blocation[1])**2)
            if(length > 0.165):
                # self.logger.error("length is longer than 165:" + str(length))
                length = 0.165
            pitch = math.degrees(math.asin(length/0.165))
            return pitch
        else:
            self.logger.error("not found key:" + key)
            raise NotFoundPlayerException("not found key:" + key)

    def getPlayerLocation(self, team=0, playerNumber=1):
        """
        if team zero means leftTeam, while one means rightTeam.
        """
        if(team == 0):
            key = "left_" + str(playerNumber)
        else:
            key = "right_" + str(playerNumber)

        location = self.playersHeadLocations.get(key)
        if(location is not None):
            return copy.copy(location)
        else:
            # time.sleep(1)
            self.logger.error("not found key:" + key)
            raise NotFoundPlayerException("not found key:" + key)
            # return None

    def getBallLocation(self):
        return copy.copy(self.ballLocation)

    def getAllPlayersLocation(self):
        return self.playersBodyLocations


class NotFoundPlayerException(RuntimeError):
    def __init__(self, arg):
        self.args = arg
