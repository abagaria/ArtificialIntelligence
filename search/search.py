# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class Node:
    # Nodes constitute a graph

    def __init__(self, path, state, priority = 1.0):

        # path is the sequence of steps to get to the current node
        self.path = path

        # state is (x, y) of the current node
        self.state = state

        # low priority is popped first from priority queues
        self.priority = priority


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    successors is of the form [(x, y), direction, cost]
    """

    # For DFS, implement fringe as LIFO stack
    fringe = util.Stack()

    return graphSearch(problem, fringe)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # For BFS, implement fringe as a FIFO queue
    fringe = util.Queue()
    return graphSearch(problem, fringe)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    priorityFunc = lambda node: node.priority
    return graphSearch(problem, fringe=util.PriorityQueueWithFunction(priorityFunc))

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    priorityFunc = lambda node : node.priority + heuristic(node.state, problem)
    return graphSearch(problem, fringe=util.PriorityQueueWithFunction(priorityFunc))

def graphSearch(problem, fringe, debug = None):

    # store all visited nodes in closed
    closed = set()

    # root of the tree
    root = Node([], problem.getStartState())

    # initialize the fringe with the root of the tree
    fringe.push(root)

    while not fringe.isEmpty():

        currentNode = fringe.pop()

        # Goal test
        if problem.isGoalState(currentNode.state):
            return currentNode.path

        # Expand unexplored nodes
        if currentNode.state not in closed:
            closed.add(currentNode.state)

            for (state, action, cost) in problem.getSuccessors(currentNode.state):
                # if debug:
                #     print "state: %s, action: %s, cost: %s" % (state.position, action, cost)
                childPath = currentNode.path + [action]
                childNode = Node(path=childPath, state=state)
                childNode.priority = cost + currentNode.priority
                fringe.push(childNode)

    # if no goal node found, return error
    return -1

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
