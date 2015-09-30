# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodDistances   = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]

        ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) \
                          for ghostState in newGhostStates]

        closestGhostDistance = min(ghostDistances)

        if closestGhostDistance < 4:
            evalFunc = -100.0

        elif closestGhostDistance < 6:
            evalFunc = 0.0

        elif not len(foodDistances):
            evalFunc = 999999.0

        else:
            evalFunc = 10.0 * successorGameState.getScore() + 1.0 / (min(foodDistances) + 1)

        return evalFunc

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        # generate the successors for every legal pacman action
        successors = [gameState.generateSuccessor(0, action) for action in gameState.getLegalActions(0)]

        # rate the successors
        initialDepth = 1
        evaluated_successors = [self.MinValue(successor, initialDepth) for successor in successors]

        best_score = evaluated_successors.index(max(evaluated_successors))

        return gameState.getLegalActions(0)[best_score]


    def MinValue(self, gameState, currentDepth):
        # base case
        if self.terminalTest(gameState, currentDepth):
            return self.evaluationFunction(gameState)

        # recursive step
        v = 9999999999.0
        agent_num = currentDepth % gameState.getNumAgents()
        numGhosts = gameState.getNumAgents() - 1

        # print agent_num

        for action in gameState.getLegalActions(agent_num):
            # iterate through the possible actions, generate a successor for the action
            successor = gameState.generateSuccessor(agent_num, action)

            # depending on which tree layer we are at, recurse to the minValue or maxValue

            if currentDepth % numGhosts == 0:
                # simulate pacman's turn
                v = min(v, self.MaxValue(successor, currentDepth + 1))

            else:
                # simulate ghosts turn
                v = min(v, self.MinValue(successor, currentDepth + 1))
        return v

    def MaxValue(self, gameState, currentDepth):

        if self.terminalTest(gameState, currentDepth):
            return self.evaluationFunction(gameState)
        # print 'max'
        v = -9999999999.0

        # number for pacman
        agent_num = 0

        for action in gameState.getLegalActions(agent_num):

            successor = gameState.generateSuccessor(agent_num, action)

            v = max(v, self.MinValue(successor, currentDepth + 1))
        return v

    def terminalTest(self, gameState, currentDepth):

        if gameState.isWin() | gameState.isLose():
            # print "current depth: ", currentDepth
            return True # if max depth is reached

        if currentDepth == (gameState.getNumAgents() * self.depth):
            # print "current depth: ", currentDepth
            return True

        # print "Terminal Score: ", self.evaluationFunction(gameState)
        return False



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

