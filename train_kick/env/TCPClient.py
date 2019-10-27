#!/usr/bin/python3

import socket
import threading
import time
import struct

from train_kick.env.Logger import Logger


class AgentConnection(threading.Thread):
    """
    This class is for connection to the agent port of Simspark server.
    """
    logger = Logger.getLogger("AgentConnection")

    serverDefaultIP = "127.0.0.1"
    agentDefaultPort = 3100
    monitorDefaultPort = 3200

    def __init__(self, id, threadName, robot):
        super(AgentConnection, self).__init__()
        self.setName(threadName + "_w")

        self.id = id
        self.threadName = threadName
        self.isStopped = False
        self.send_buffer = ""
        self.monitor_send_buffer = ""

        self.robot = robot

    def __str__(self):
        return "Server IP:" + self.server + " Server agent port:" + str(
            self.port)

    def connectServers(self,
                       server=serverDefaultIP,
                       port=agentDefaultPort,
                       mport=monitorDefaultPort):
        self.server = server
        self.port = port
        self.mport = mport

        cstatus = True
        while(cstatus):
            try:
                address = (self.server, self.port)
                maddress = (self.server, self.mport)
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(address)

                self.msock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.msock.connect(maddress)

                self.rethread = ReceivingThread(self.threadName + "_r", self.sock,
                                                self.isStopped, self.robot)
                self.rethread.start()

                # For monitor thread
                self.mrethread = MonitorReceivingThread(self.threadName + "_r_m",
                                                        self.msock, self.isStopped, self.robot)
                self.mrethread.start()

                cstatus = False
            except socket.error as e:
                self.logger.exception("socket.error" + repr(e))
                self.logger.error(str(self.server) + ":" + str(self.port))
                self.close()
                time.sleep(30)
                self.isStopped = False

    def close(self):
        self.isStopped = True
        time.sleep(1)
        if(hasattr(self, 'rethread')):
            self.rethread.close()

        if(hasattr(self, 'mrethread')):
            self.mrethread.close()

        time.sleep(1)
        if(hasattr(self, 'sock')):
            self.sock.close()
        if(hasattr(self, 'msock')):
            self.msock.close()

    def run(self):
        try:
            self.logger.info("begin to send message...")
            next_time = time.time() + 0.02
            while (not self.isStopped):
                if (self.send_buffer != ""):
                    self.logger.debug("Send content:" + self.send_buffer)
                    self.__socketSend(self.send_buffer+"(syn)")
                    self.send_buffer = ""
                else:
                    self.logger.debug("Send content:(syn)")
                    self.__socketSend("(syn)")

                # send monitor message
                if (self.monitor_send_buffer != ""):
                    self.logger.debug("Send monitor content:" + self.monitor_send_buffer)
                    self.__monitorSend(self.monitor_send_buffer)
                    self.monitor_send_buffer = ""
                else:
                    self.logger.debug("Send monitor content: (reqfullstate)")
                    self.__monitorSend("(reqfullstate)")

                after_action = time.time()
                sleeptime = next_time - after_action
                if (sleeptime > 0):
                    next_time += 0.02
                    self.logger.debug("sleeptime" + str(sleeptime))
                    if (sleeptime > 0.01):
                        time.sleep(sleeptime - 0.01)
                    else:
                        time.sleep(0.001)
                else:
                    next_time = after_action + 0.02
                    self.logger.debug("adjust time" + str(abs(sleeptime)))
                    time.sleep(0.015)

        except socket.error as e:
            self.logger.exception("socket.error" + repr(e))
            self.sock.close()
            exit(1)

    def sendMessage(self, message):
        self.send_buffer = message

    def sendMonitorMessage(self, message):
        self.monitor_send_buffer = message

    def __socketSend(self, message):
        msg = bytes(message, 'ascii')
        self.sock.send(struct.pack("!I", len(msg)) + msg)

    def __monitorSend(self, message):
        msg = bytes(message, 'ascii')
        self.msock.send(struct.pack("!I", len(msg)) + msg)


class ReceivingThread(threading.Thread):

    logger = Logger.getLogger("ReceivingThread")

    def __init__(self, name, sock, isStopped, robot):
        super(ReceivingThread, self).__init__()
        self.setName(name)

        self.name = name
        self.sock = sock
        self.isStopped = isStopped
        self.logger.info(name + " receive init")

        self.robot = robot
        self.lastReadTime = time.time()

    def run(self):
        while (not self.isStopped):
            message = self.readMessage()
            # self.logger.info("Receive:"+message)

            self.udpateRobot(message)
            self.lastReadTime = time.time()

    def close(self):
        self.isStopped = True

    def readMessage(self):
        length_no = self.sock.recv(4, socket.MSG_WAITALL)
        length = struct.unpack("!I", length_no)
        msg = self.sock.recv(length[0], socket.MSG_WAITALL)
        message = msg.decode('ascii')
        return message

    def udpateRobot(self, message):
        self.logger.debug(message)
        self.robot.update(message)
        # self.logger.debug(self.robot.getAllStates())


class MonitorReceivingThread(threading.Thread):

    logger = Logger.getLogger("MonitorReceivingThread")

    def __init__(self, name, sock, isStopped, robot):
        super(MonitorReceivingThread, self).__init__()
        self.logger.info(name + " receive init")

        self.setName(name)
        self.name = name
        self.sock = sock
        self.isStopped = isStopped
        self.robot = robot

        self.lastReadTime = time.time()

    def run(self):
        while (not self.isStopped):
            message = self.readMessage()
            # self.logger.info("Receive:"+message)

            self.udpateGrandStates(message)
            self.lastReadTime = time.time()

    def close(self):
        self.isStopped = True

    def readMessage(self):
        length_no = self.sock.recv(4, socket.MSG_WAITALL)
        length = struct.unpack("!I", length_no)
        msg = self.sock.recv(length[0], socket.MSG_WAITALL)
        message = msg.decode('ascii')
        return message

    def udpateGrandStates(self, message):
        self.logger.debug(message)
        self.robot.grandEnvs.updateGrandStates(message)
        # self.logger.debug(GrandEnvs.getPlayerLocation(0, 1))
        # self.logger.debug("ball: " + str(GrandEnvs.ballLocation))
        # self.logger.debug("players: " + str(GrandEnvs.getAllPlayersLocation()))
