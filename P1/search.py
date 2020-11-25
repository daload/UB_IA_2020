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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # Declaramos el stack
    stack = util.Stack()
    stack.push(((problem.getStartState(), ''), []))
    # Creamos la lista de nodos visitados
    visitados = []

    while not stack.isEmpty():
        # Cogemos el primer elemento del stack
        actual = stack.pop()
        # Si el nodo actual es la meta podemos devolcer la lista de acciones
        if problem.isGoalState(actual[0][0]):
            return actual[1]
        visitados.append(actual[0][0])  # Añadimos el nodo actual a la lista de  nodos visitados
        hijos = []  # Lista de hijos del nodo actual
        for hijo in problem.getSuccessors(actual[0][0]):
            # Si el hijo no esta en la lista de nodos visitados lo añadimos a la lista de hijos
            if hijo[0] not in visitados:
                hijos.append((hijo, actual[1] + [hijo[1]]))

        # Añadimos los hijos por visitar al stack
        for nodo in hijos:
            stack.push(nodo)
    # Si no encuentra solucion devuelve una lista vacia
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # Declaramos la cola
    queue = util.Queue()
    queue.push(((problem.getStartState(), ''), []))
    # Creamos la lista de nodos visitados
    visitados = [problem.getStartState()]

    while not queue.isEmpty():
        # Cogemos el primer elemento del stack
        current = queue.pop()
        # Si el nodo actual es la meta podemos devolcer la lista de acciones
        if problem.isGoalState(current[0][0]):
            return current[1]
        # visitados.append(current[0][0])

        for hijo in problem.getSuccessors(current[0][0]):
            if hijo[0] not in visitados:
                # Añadimos el hijo a la lista de visitados para evitar repeticiones
                visitados.append(hijo[0])
                # Añadimos el hijo al queue para visitarlo
                queue.push((hijo, current[1] + [hijo[1]]))

    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Declaramos el heap y le añadimos el nodo inicial
    heap = util.PriorityQueue()
    heap.push(((problem.getStartState(), ''), [], 0), 0)
    visited = {problem.getStartState(): 0}

    while not heap.isEmpty():
        # Cogemos el primer elemento del heap
        current = heap.pop()
        if problem.isGoalState(current[0][0]):
            # Si el nodo actual es la meta podemos devolcer la lista de acciones
            return current[1]

        for hijo in problem.getSuccessors(current[0][0]):
            # Si el hijo no esta en visitados, lo añadimos al heap
            if hijo[0] not in visited:
                heap.push((hijo, current[1] + [hijo[1]], current[2] + hijo[2]),
                          current[2] + hijo[2] + heuristic(hijo[0], problem))
            # Si el hijo esta en visitados pero el coste es menor actualizamos el heap
            elif (current[2] + hijo[2]) < visited[hijo[0]]:
                heap.update((hijo, current[1] + [hijo[1]], current[2] + hijo[2]),
                          current[2] + hijo[2] + heuristic(hijo[0], problem))
            # Si el hijo esta en visitados pero el coste es mayor al que ya tenemos pasamos a la siguiente iteracion
            else:
                continue

            # Actualizamos la lista de visitados
            visited[hijo[0]] = (current[2] + hijo[2])

    return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
