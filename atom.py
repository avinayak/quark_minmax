from __future__ import division

from copy import deepcopy
from functools import reduce
import operator
import numpy as np
import quark
import rle
import os
import time
import random
import numba

class Atom():
    def __init__(self):
        self.board = rle.encode(np.zeros((10, 8), np.int8))
        self.currentPlayer = 1

    def getPossibleActions(self):
        newBoard = rle.decode(self.board, (10, 8))
        moves = quark.get_legal_cell_locs(newBoard, self.currentPlayer)
        return [Action(player=self.currentPlayer, x=x, y=y) for x,y in moves]

    def takeAction(self, action):
        newState = deepcopy(self)
        newBoard = rle.decode(newState.board, (10, 8))
        newBoard = quark.play(newBoard, action.x, action.y, action.player)
        newState.currentPlayer = self.currentPlayer * -1
        newState.board = rle.encode(newBoard)
        return newState

    def isTerminal(self):
        return quark.is_gameover(rle.decode(self.board, (10, 8)))

    def getReward(self):
        newBoard = rle.decode(self.board, (10, 8))
        if quark.is_gameover(newBoard):
            degrees = quark.get_total_degrees_by_color(newBoard)
            if degrees[1] == 0:
                return -1*(320 - degrees[-1])**2
            else:
                return 1*(320-degrees[1])**2
        return False


class Action():
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y

    def __str__(self):
        return str((self.x, self.y))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.player == other.player

    def __hash__(self):
        return hash((self.x, self.y, self.player))