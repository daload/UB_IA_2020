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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        newFood = successorGameState.getFood()  # Grid of foods (T = True, F = False)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]  # Secs to finish power pellet
        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()  # Define score
        # Lists of manhattan distances with food and ghosts
        food_list = [manhattanDistance(d, newPos) for d in newFood.asList()]
        ghosts = [manhattanDistance(d, newPos) for d in successorGameState.getGhostPositions()]
        # If there is no food, the game is over so we set recp_food to 1 to avoid divisions by 0
        if len(food_list) != 0:
            recp_food = 1 / min(food_list)
        else:
            recp_food = 1
        # Penalization of "Stop" action
        if action == "Stop":
            score = score / 10
        # If the distance with a ghost is below 2, we set score to a low value to not take that action
        if min(ghosts) < 2:
            score = float("-inf")
        return score + recp_food


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        "*** YOUR CODE HERE ***"
        # Return de 2nd element of the tuple returned by max_value (value, action)
        # We return de 2nd item of the tuple that is the action to make.
        depth = self.depth * gameState.getNumAgents()
        return self.max_v(gameState, depth, 0, 1)[1]

    def minimax(self, game, depth, agent):
        # If we arrive to a terminal node return de score
        if depth == 0 or game.isLose() or game.isWin():
            return self.evaluationFunction(game)
        # Calculate next agent
        next_agent = 0 if agent == game.getNumAgents() - 1 else agent + 1
        # MAX layer for Pacman, MIN layer for ghosts
        # Return of the desired value
        return self.max_v(game, depth, agent, next_agent)[0] if agent == 0 else \
        self.min_v(game, depth, agent, next_agent)[0]

    def max_v(self, game, depth, agent, next_agent):
        # Calculate all the values for each action
        values = [(self.minimax(game.generateSuccessor(agent, action), depth - 1, next_agent), action) for
                  action in game.getLegalActions(agent)]
        # Return the tuple of the highest value and its action
        return max(values)

    def min_v(self, game, depth, agent, next_agent):
        # Calculate all the values for each action
        values = [(self.minimax(game.generateSuccessor(agent, action), depth - 1, next_agent), action) for
                  action in game.getLegalActions(agent)]
        # Return the tuple of the lowest value and its action
        return min(values)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Depth, alpha and beta declaration
        depth = self.depth * gameState.getNumAgents()
        alpha, beta = float("-inf"), float("inf")
        # return 2nd element of max_v (max_v returns a tuple of the score and the action)
        return self.max_v(gameState, depth, 0, 1, alpha, beta)[1]

    def minimax_alphabeta(self, game, depth, agent, a, b):
        # If we arrive to a terminal node return de score
        if depth == 0 or game.isLose() or game.isWin():
            return self.evaluationFunction(game)
        # Calculate next agent
        next_agent = 0 if agent == game.getNumAgents() - 1 else agent + 1
        # MAX layer for Pacman, MIN layer for ghosts
        # Return of the desired value
        return self.max_v(game, depth, agent, next_agent, a, b)[0] if agent == 0 else \
        self.min_v(game, depth, agent, next_agent, a, b)[0]

    def max_v(self, game, depth, agent, next_agent, a, b):  # a = alpha, b = beta
        # same as max_v of minimax but with pruning
        v = (float("-inf"), "")
        for action in game.getLegalActions():
            v = max(v, (
            self.minimax_alphabeta(game.generateSuccessor(agent, action), depth - 1, next_agent, a, b), action))
            # Pruning
            if v[0] > b:
                return v
            else:
                a = max(a, v[0])
        return v

    def min_v(self, game, depth, agent, next_agent, a, b):
        # Same as min_v of minimax but with pruning
        v = (float("inf"), "")
        for action in game.getLegalActions(agent):
            v = min(v, (self.minimax_alphabeta(game.generateSuccessor(agent, action), depth - 1, next_agent, a, b), action))
            # Pruning
            if v[0] < a:
                return v
            else:
                b = min(b, v[0])
        return v


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
        depth = self.depth * gameState.getNumAgents()
        return self.max_v(gameState, depth, 0, 1)[1]

    def expectimax(self, game, depth, agent, probability):
        # If we arrive to a terminal node return de score
        if depth == 0 or game.isLose() or game.isWin():
            return self.evaluationFunction(game) * probability
        # Calculate next agent
        next_agent = 0 if agent == game.getNumAgents() - 1 else agent + 1
        # MAX layer for Pacman, Expect value for ghosts
        # Return of the desired value
        return self.max_v(game, depth, agent, next_agent)[0] if agent == 0 else \
            self.expect_v(game, depth, agent, next_agent)[0]

    def max_v(self, game, depth, agent, next_agent):
        # Calculate all the values for each action
        values = [(self.expectimax(game.generateSuccessor(agent, action), depth - 1, next_agent, 1), action) for
                  action in game.getLegalActions(agent) if action != "Stop"]
        # Return the tuple of the highest value and its action
        return max(values)

    def expect_v(self, game, depth, agent, next_agent):
        prob = 1 / len(game.getLegalActions(agent))
        value = [0, ""]
        for action in game.getLegalActions(agent):
            if action != "Stop":
                val = (self.expectimax(game.generateSuccessor(agent, action), depth - 1, next_agent, prob), action)
                value[1] = val[1]
                value[0] += val[0]*prob

        return value


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: With this evaluation function we will compensate for being able to eat ghosts, eating more food,
    being close to a ghost if we can eat him and being close to food while we will be deducting score for all the
    food that is still on the board, and being close to ghosts if we can't eat their souls
    """
    "*** YOUR CODE HERE ***"
    # Useful info
    new_pos = currentGameState.getPacmanPosition()
    new_food = currentGameState.getFood().asList()
    new_ghost_states = currentGameState.getGhostStates()
    scared_times = [ghostState.scaredTimer for ghostState in new_ghost_states]
    new_pellets = currentGameState.getCapsules()

    # Get all foods (normal food + power pellets)
    all_food = new_pellets + new_food
    # Get the distances with all foods
    food_dist = [manhattanDistance(d, new_pos) for d in all_food]
    # Lowest time to eat a ghost
    lowest_party_time = min(scared_times)
    # Distances with ghosts
    ghosts_distances = [manhattanDistance(d, new_pos) for d in currentGameState.getGhostPositions()]

    # score will start with the game score - the reciprocal value of the food to be eaten, if there is no food left we
    # will just add 2 to avoid errors
    score = currentGameState.getScore() - 1/len(all_food) if len(all_food) != 0 else currentGameState.getScore() + 2

    # We add the reciprocal value of the closest food, and add 2 if there is no food left to avoid errors
    score += 1/min(food_dist) if len(food_dist) != 0 else 2

    # If we eat a power pellet we add the reciprocal value of the distance to the closest ghost, and add 2 if the
    # distance to the closest ghost is 0 to avoid errors. Also, we add the reciprocal value of te highest timer to
    # give more weight to the fact that we are gods.
    # If we can't eat ghosts we deduct the reciprocal value of the distance to the closes ghost and add 2
    # if the distance to the closest ghost is 0 to avoid errors
    if lowest_party_time == 0:
        score -= 1/ min(ghosts_distances) if min(ghosts_distances) != 0 else 2
    else:
        score += 1 / min(ghosts_distances) if min(ghosts_distances) != 0 else 2
        score += 1 / max(scared_times)
    return score

# Abbreviation
better = betterEvaluationFunction
