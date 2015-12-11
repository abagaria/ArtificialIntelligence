# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Agent
import game
from util import manhattanDistance
from math import log10

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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


def betterEvaluationFunction(self, currentGameState):
        global eaten
        newPos = currentGameState.getPacmanPosition()
        newFood = currentGameState.getFood()
        newGhostStates = currentGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        powerPellets = currentGameState.getCapsules()

        foodDistances   = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]

        ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) \
                          for ghostState in newGhostStates]

        closestGhostDistance = min(ghostDistances)

        if closestGhostDistance < 1:
            evalFunc = -100.0

        # elif closestGhostDistance < 2:
        #     evalFunc = 0.0

        elif not len(foodDistances):
            evalFunc = 999999.0

        else:
            evalFunc = 10.0 * currentGameState.getScore() + 1.0 / (min(foodDistances) + 1)

        # If you are close to a power pellet, go get it
        if (len(powerPellets)) < 2 and not eaten :
            evalFunc += 100000.0
            eaten = not eaten

        # If pacman has already eaten the power pellet
        if newScaredTimes:
            evalFunc += (2.0 / (min(ghostDistances) + 1.0))

        return evalFunc


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.initialFood = len(self.getFood(gameState).asList())

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    actions = gameState.getLegalActions(self.index)
    # actions.remove(Directions.STOP)
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
    if pos != util.nearestPoint(pos):
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

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    actions = gameState.getLegalActions(self.index)
    # actions.remove(Directions.STOP)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    for idx,a in enumerate(actions):
        baby = self.getSuccessor(gameState, a)
        qsum = [self.evaluate(baby, action) for action in baby.getLegalActions(self.index)]
        values[idx] += min(qsum) 

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
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    distToGhost = self.avoidGhosts(gameState, action)
    features['distanceToClosestGhost'] = distToGhost
    # print "contribution from nearest ghost = ", -10 * distToGhost

    features['inCorner'] = self.avoidCorners(gameState, action)

    # if close to a power capsule, eat it
    #features['distanceToPowerCap'] = self.closeToPowerPellet(gameState, action)

    # If power pelleted, then attack the ghosts!
    # features['attackTheGhosts'] = self.attackGhosts(gameState, action)

    # if features['attackTheGhosts'] != 0.0:
        # features['distanceToClosestGhost'] = 0

    features['numGhosts'] = len(ghosts)

    # if you are a ghost, then approach pacman and try to eat it
    # features['attackPacman'] = self.attackPacman(gameState, action)

    if (self.initialFood - len(foodList)) > 3:
        # minimize difference to our side of board
        features['returnHome'] = self.getDistanceToMySide(gameState, action)
    else:
        features['returnHome'] = 0.0

    prevState = self.getPreviousObservation()
    nextState = self.getCurrentObservation()
    if prevState:
        if self.getScore(prevState) < self.getScore(nextState):
            self.initialFood = len(foodList)
            print "initial food = ", self.initialFood


    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'distanceToClosestGhost': -150, 'inCorner' : -150, \
            'distanceToPowerCap': 10, 'attackTheGhosts': -8, 'numGhosts': -100, 'attackPacman': 200, 'returnHome': -5}


  def getDistanceToMySide(self, gameState, action):
     successor = self.getSuccessor(gameState, action)
     myFood = self.getFoodYouAreDefending(gameState).asList()

     if len(myFood) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in myFood])

      return minDistance


  # return the distance from pacman to the closest ghost
  def avoidGhosts(self, gameState, action):

    successor = self.getSuccessor(gameState, action)
    minDistance = self.getDistToClosestGhost(successor)

    if minDistance >= 3:
        return 0

    return 1.0 / (minDistance + 1.0)

  ###
  def getDistToClosestGhost(self, successor):
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

      ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

      myPos = successor.getAgentState(self.index).getPosition()

      if len(ghosts) > 0:
          dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
          return min(dists)

      return 0

  # Avoid 3 edged corners because pacman often gets stuck in those
  def avoidCorners(self, gameState, action):

      successor = self.getSuccessor(gameState, action)

      legalActions = successor.getLegalActions(self.index)

      # Are we in a corner?
      isPacman = successor.getAgentState(self.index).isPacman
      inCorner = len(legalActions) == 2
      ghostNearBy = self.getDistToClosestGhost(successor) <= 5

      if  inCorner and isPacman and ghostNearBy:
          print "I am in a corner :O"
          return True
      # print "legalActions=", legalActions
      return False

  # if there is a power pellet close to you, return the distance to it
  # else, return 0
  def closeToPowerPellet(self, gameState, action):

      successor = self.getSuccessor(gameState, action)

      # get the location of all the power pellets
      power_pellets = successor.getCapsules()

      myPos = successor.getAgentState(self.index).getPosition()

      dists = []
      if len(power_pellets) >= 1:
        dists = [self.getMazeDistance(myPos, powerpos) for powerpos in power_pellets]

      minDistanceToPowerPellet = 0
      if dists:
        minDistanceToPowerPellet = min(dists)

      if successor.getAgentState(self.index).isPacman:
        return 1.0 / (minDistanceToPowerPellet + 1.0)
      return 0

  def attackGhosts(self, gameState, action):
      successor = self.getSuccessor(gameState, action)
      distClosestGhost = self.getDistToClosestGhost(successor)
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]

      scaredTime = max([ghostState.scaredTimer for ghostState in invaders])

      isPacman = successor.getAgentState(self.index).isPacman
      toChase =  scaredTime >= distClosestGhost
      if isPacman and toChase:
        return 1.0 / (distClosestGhost + 1.0)
      return 0.0

  # def attackPacman(self, gameState, action):
  #     successor = self.getSuccessor(gameState, action)
  #     isGhost = not successor.getAgentState(self.index).isPacman
  #     print 'is ghost=', isGhost
  #     if isGhost:
  #       enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
  #       invaders = [a for a in enemies if a.isPacman]
  #       myState = successor.getAgentState(self.index)
  #       myPos = myState.getPosition()
  #       print "my position = ", myPos
  #       print "invaders = ", invaders
  #
  #       dists = []
  #       if len(invaders) > 0:
  #         dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
  #       print "distances=", dists
  #
  #       if len(dists) > 0:
  #         if min(dists) == 1:
  #           return 99999
  #         return 1.0 / (min(dists) + 1.0)
  #     return 0


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
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}