# """
# In search.py, you will implement generic search algorithms which are called by
# Pacman agents (in searchAgents.py).

# Please only change the parts of the file you are asked to.  Look for the lines
# that say

# "*** YOUR CODE HERE ***"

# Follow the project description for details.

# Good luck and happy searching!
# """

# import util

# class SearchProblem:
#     """
#     This class outlines the structure of a search problem, but doesn't implement
#     any of the methods (in object-oriented terminology: an abstract class).

#     You do not need to change anything in this class, ever.
#     """

#     def getStartState(self):
#         """
#         Returns the start state for the search problem.
#         """
#         util.raiseNotDefined()

#     def isGoalState(self, state):
#         """
#           state: Search state

#         Returns True if and only if the state is a valid goal state.
#         """
#         util.raiseNotDefined()

#     def getSuccessors(self, state):
#         """
#           state: Search state

#         For a given state, this should return a list of triples, (successor,
#         action, stepCost), where 'successor' is a successor to the current
#         state, 'action' is the action required to get there, and 'stepCost' is
#         the incremental cost of expanding to that successor.
#         """
#         util.raiseNotDefined()

#     def getCostOfActions(self, actions):
#         """
#          actions: A list of actions to take

#         This method returns the total cost of a particular sequence of actions.
#         The sequence must be composed of legal moves.
#         """
#         util.raiseNotDefined()

# def tinyMazeSearch(problem):
#     """
#     Returns a sequence of moves that solves tinyMaze.  For any other maze, the
#     sequence of moves will be incorrect, so only use this for tinyMaze.
#     """
#     from game import Directions
#     s = Directions.SOUTH
#     w = Directions.WEST
#     print("Solution:", [s, s, w, s, w, w, s, w])
#     return  [s, s, w, s, w, w, s, w]

# def depthFirstSearch(problem: SearchProblem):
#     """
#     Search the deepest nodes in the search tree first.

#     Your search algorithm needs to return a list of actions that reaches the
#     goal. Make sure to implement a graph search algorithm.

#     To get started, you might want to try some of these simple commands to
#     understand the search problem that is being passed in:

#     print("Start:", problem.getStartState())
#     print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
#     print("Start's successors:", problem.getSuccessors(problem.getStartState()))
#     """
#     "*** YOUR CODE HERE ***"

#     util.raiseNotDefined()

# def breadthFirstSearch(problem: SearchProblem):
#     """Search the shallowest nodes in the search tree first."""
#     "*** YOUR CODE HERE ***"
#     util.raiseNotDefined()

# def uniformCostSearch(problem: SearchProblem):
#     """Search the node of least total cost first."""
#     "*** YOUR CODE HERE ***"
#     util.raiseNotDefined()

# def nullHeuristic(state, problem=None):
#     """
#     A heuristic function estimates the cost from the current state to the nearest
#     goal in the provided SearchProblem.  This heuristic is trivial.
#     """
#     return 0

# def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
#     """Search the node that has the lowest combined cost and heuristic first."""
#     "*** YOUR CODE HERE ***"
#     util.raiseNotDefined()


# # Abbreviations
# bfs = breadthFirstSearch
# dfs = depthFirstSearch
# astar = aStarSearch
# ucs = uniformCostSearch




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

def depthFirstSearch(problem:SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    stack = [(problem.getStartState(), [])]  
    visited = set()  

    while stack:
        state, actions = stack.pop()  

        if state in visited:  
            continue
        visited.add(state)  

        if problem.isGoalState(state):  
            return actions  

        successors = problem.getSuccessors(state)  
        for next_state, action, _ in successors:
            new_actions = actions + [action] 
            stack.append((next_state, new_actions))  

    return []  
    
def breadthFirstSearch(problem:SearchProblem):
    """Search the shallowest nodes in the search tree first."""

    queue = [(problem.getStartState(), [])]
    visited = set()

    while queue:
        state, actions = queue.pop(0)

        if state in visited:
            continue
        visited.add(state)

        if problem.isGoalState(state):
            return actions
        
        successors = problem.getSuccessors(state)
        for next_state, action, i in successors:
            new_actions = actions + [action]
            queue.append((next_state, new_actions))
    return []

def uniformCostSearch(problem:SearchProblem):
    """Search the node of least total cost first."""
    front = util.PriorityQueue()
    trace = {}
    start_state = problem.getStartState()
    front.push((start_state, [], 0), 0)
    trace[start_state] = (None, None, 0)

    while not front.isEmpty():
        current, actions, curr_cost = front.pop()
        if problem.isGoalState(current):
            return actions
        successors = problem.getSuccessors(current)
        for next, action, step_cost in successors:
            total_cost = curr_cost + step_cost
            if next not in trace or total_cost < trace[next][2]:
                trace[next] = (current, action, total_cost)
                front.update((next, actions + [action], total_cost), total_cost)
    return []




def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem:SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    front = util.PriorityQueue()
    start_state = problem.getStartState()
    start_heuristic = heuristic(start_state, problem)
    visited = []
    front.push( (start_state, [], 0), start_heuristic)
    directions = []
    while not front.isEmpty():
        get_xy, directions, get_cost = front.pop()

        if problem.isGoalState(get_xy):
            return directions
        
        if not get_xy in visited:
            # Track visited_nodes
            visited.append(get_xy)
            
            for coordinates, direction, successor_cost in problem.getSuccessors(get_xy):
                if not coordinates in visited:
                    # Pass by reference
                    actions_list = list(directions)
                    actions_list += [direction]
                    # Get cost so far
                    cost_actions = problem.getCostOfActions(actions_list)
                    get_heuristic = heuristic(coordinates, problem)
                    front.push( (coordinates, actions_list, 1), cost_actions + get_heuristic)
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch