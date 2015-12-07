# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a self to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import itertools

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.setWeights(None)
    self.setCenter(gameState)
    self.label_depth(gameState)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  def setWeights(self, weights):
      pass


  def setCenter(self, gameState):
    grid = gameState.getWalls()
    center_x, center_y = grid.width / 2, grid.height / 2


    grid = grid.data
    if not grid[center_x][center_y]:
        self.center = (center_x, center_y)
    elif not grid[center_x][center_y + 1]:
        self.center = (center_x, center_y + 1)
    elif not grid[center_x][center_y - 1]:
        self.center = (center_x, center_y - 1)
    else:
        print 'CRITICAL setCenter ERROR'

  def label_depth(self, gameState):
      self.grid = []
      self.depth_grid = []
      grid = gameState.getWalls().data
      for i in range(len(grid)):
          self.grid.append([-1] * len(grid[i]))
          self.depth_grid.append([-1] * len(grid[i]))

      # dead_end_filling(grid)
      # dead_end_connecting(grid)
      self.DFS_DEAD_END(self.center, self.center, grid)
      self.recalculate_depth(self.center, grid, 0)

  def recalculate_depth(self, current_position, walls, depth):
      x,y = current_position
      if self.depth_grid[x][y] >= 0:
          return

      self.depth_grid[x][y] = depth

      options = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
      for ox, oy in options:
          if walls[ox][oy]:
              continue
          new_depth = depth
          if self.grid[ox][oy] > 0:
              new_depth += 1

          self.recalculate_depth((ox,oy), walls, new_depth)

  def DFS_DEAD_END(self, current_position, last_position, walls):
      x,y = current_position
      
      if self.grid[x][y] >= 0:
          #visited before and not a dead end
          return self.grid[x][y]

      self.grid[x][y] = 0
      options = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
      num_paths = 0
      dead_end_depth = 0
      for ox, oy in options:
          if walls[ox][oy]:
              continue
          if (ox,oy) == last_position:
              continue
          dead_endness = self.DFS_DEAD_END((ox,oy), current_position, walls)
          if dead_endness == 0:
              num_paths += 1
          dead_end_depth = max(dead_endness, dead_end_depth)

      if num_paths < 1:
          self.grid[x][y] = 1 + max(dead_end_depth, 1)
          
      return self.grid[x][y]

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myPos = successor.getAgentState(self.index).getPosition()
    foodList = self.getFood(successor).asList()    

    safeFoodList = [food for food in foodList if self.depth_grid[int(food[0])][int(food[1])] <= 2]

    features['food'] = -len(foodList)#self.getScore(successor)
    features['safeFood'] = -len(safeFoodList)

    # Minimum distance to enemy
    minDistance = min([self.getMazeDistance(myPos, enemy) for enemy in [successor.getAgentPosition(i) for i in self.getOpponents(gameState)]])
    features['distanceToEnemy'] = minDistance

    # Enemy within 1 square
    features['enemyWithinOneSquare'] = minDistance == 1

    # Score
    features['score'] = self.getScore(successor)

    # Distance to center
    features['distanceToCenter'] = self.getMazeDistance(self.center, myPos)

    # Deadend depth
    features['deadendDepth'] = self.depth_grid[int(myPos[0])][int(myPos[1])]

    # Number of capsules
    capsuleList = self.getCapsules(successor)
    features['capsules'] = -len(capsuleList)

    # Compute distance to the nearest capsule
    if len(capsuleList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in capsuleList])
      features['distanceToCapsule'] = minDistance

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    if len(safeFoodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in safeFoodList])
      features['distanceToSafeFood'] = minDistance

    return features

  def getWeights(self, gameState, action):
    return self.weights

  def setWeights(self, yo):
    self.weights = {'capsules':
151.2,
'distanceToCapsule':
-0.95,
'food':
123.9,
'distanceToFood':
-0.38,
'deadendDepth':
-0.219,
'enemyWithinOneSquare':
-155.3,
'distanceToEnemy':
0.26,
'distanceToCenter':
-0.859,
'score':
452.0,
'safeFood':
34.2,
'distanceToSafeFood':
-2.97}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
  
    capsuleList = self.getCapsulesYouAreDefending(successor)
    features['ownCapsules'] = -len(capsuleList)
    features['ownFood'] = len(self.getFoodYouAreDefending(successor).asList())
    
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    features['distanceToCenter'] = self.getMazeDistance(self.center, myPos)

    features['isAfraid'] = successor.getAgentState(self.index).scaredTimer
    features['distanceToEnemy'] = min([self.getMazeDistance(myPos, successor.getAgentState(a).getPosition()) for a in self.getOpponents(successor)])

    return features

  def getWeights(self, gameState, action):
    return self.weights

  def setWeights(self, yo):
    self.weights = {'onDefense':
210.15,
'numInvaders':
-361.88,
'invaderDistance':
-53.16,
'stop':
-100.2,
'reverse':
-1.62,
'ownCapsules':
100.0,
'ownFood':
14.95,
'distanceToCenter':
26.65,
'isAfraid':
-268.0,
'distanceToEnemy':
-79.5}
