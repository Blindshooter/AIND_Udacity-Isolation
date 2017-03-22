"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def heuristics1(game, player, weight = 3):

    # Proposed heuristics is elaboration of Improved Score idea
    # It adds weight to opponent moves and
    # scales the result based on the filled spaces at the time of calculation


    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    total_spaces = float(game.width * game.height)
    free_spaces = total_spaces - float(len(game.get_blank_spaces()))

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(own_moves - weight*opp_moves)*(total_spaces-free_spaces)


def heuristics2(game, player):
    # The idea that it is better to search moves
    # in the center of the board rather near its boundaries
    # Difference in distance to center between player and opponent
    # is added as additional value to Improved Score heuristic

    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    total_spaces = float(game.width * game.height)
    free_spaces = total_spaces - float(len(game.get_blank_spaces()))

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    center_row, center_col = game.width//2, game.height//2

    row, col = game.get_player_location(player)
    opp_row, opp_col = game.get_player_location(game.get_opponent(player))

    distance = ((center_row-row)**2+(center_col-col)**2)**0.5
    opp_distance = ((center_row-opp_row)**2+(center_col-opp_col)**2)**0.5

    return (distance - opp_distance) + (own_moves - opp_moves)

def heuristics3(game, player, weight = 3.):
    # This heuristics goes one level deeper and
    # explore possible moves on the next level of moves
    # The difference of explored moves for a player and for an opponent -
    # is the value of evaluation function

    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    total_spaces = float(game.width * game.height)
    free_spaces = total_spaces - float(len(game.get_blank_spaces()))

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    #exploratory
    if float(free_spaces)/float(total_spaces)<0.3:
        future_moves = [game.forecast_move(move) for move in game.get_legal_moves(player)]
        explore_moves = 0
        for future_state in future_moves:
            new_moves = len(future_state.get_legal_moves(future_state.inactive_player))
            explore_moves = max(explore_moves, new_moves)

        return explore_moves
    else:
        return float(own_moves - weight*opp_moves)*(total_spaces-free_spaces)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return heuristics1(game, player)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left


        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        direction = (-1,-1)

        if len(legal_moves)==0:
            return direction
        else:
            direction = random.choice(legal_moves)

        #directions = []

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            # checking whether we are doing iterative deepening
            if self.iterative:
                depth = 0
                while True:
                    if self.method == 'minimax':
                        score, direction = self.minimax(game, depth)
                    elif self.method == 'alphabeta':
                        score, direction = self.alphabeta(game, depth)
                    else:
                        break
                    if direction is (-1, -1):
                        break
                    depth += 1

            else:
                if self.method == 'minimax':
                    score, direction = self.minimax(game, self.search_depth)
                elif self.method == 'alphabeta':
                    score, direction = self.alphabeta(game, self.search_depth)

            # Return the best move from the last completed search iteration

        except Timeout:
            return direction

        return direction

    def terminal_state(self, game, depth):
        # Check terminal state.
        # Used in minimax and alpha-beta pruning algorithms implementation
        # Idea of implementation through the definition of terminal state lays here:
        # https://github.com/aimacode/aima - python

        if len(game.get_legal_moves()) == 0 or depth == 0:
            return True
        return False

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()


        # Initialization
        direction = (-1,-1)
        best_score = float("-inf")

        # Minimax implementation
        # Uses support functions below
        if self.terminal_state(game, depth):
            return best_score, direction

        # Check eligible nodes and choose max value.
        for next_move in game.get_legal_moves():
            new_node = game.forecast_move(next_move)
            new_score = self.min_value(new_node, depth-1)
            if new_score > best_score:
                best_score = new_score
                direction = next_move

        return best_score, direction

    def max_value(self, game, depth):
        # Get max value for minimax algorithm

        if self.terminal_state(game, depth):
            return self.score(game, self)

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        max_score = float("-inf")

        for next_move in game.get_legal_moves():
            new_node = game.forecast_move(next_move)
            new_score = self.min_value(new_node, depth-1)
            if new_score > max_score:
                max_score = new_score

        return max_score

    def min_value(self, game, depth):
        # Get min value for minimax algorithm
        if self.terminal_state(game, depth):
            return self.score(game, self)

        # If the time lefts less than TIMER_THRESHOLD,
        # then immediately return the best move so far.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        min_score = float("inf")

        for next_move in game.get_legal_moves():
            new_node = game.forecast_move(next_move)
            new_score = self.max_value(new_node, depth-1)
            if new_score < min_score:
                min_score = new_score

        return min_score



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Initialization
        best_score = float("-inf")
        direction = (-1, -1)

        # Check for terminal state.
        if self.terminal_state(game, depth):
            return best_score, direction

        # Alphabeta pruning implementation.
        # Uses support functions below
        best_score, direction = self.alphabeta_max(game, depth, alpha, beta)

        return best_score, direction

    def alphabeta_max(self, game, depth, alpha, beta):
        # Get minimum score in Alpha-Beta pruning algorithm.

        if self.terminal_state(game, depth):
            return self.score(game, self), game.get_player_location(self)

        # Do not forget about time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        max_score = float("-inf")
        direction = (-1, -1)

        for next_move in game.get_legal_moves():
            new_node = game.forecast_move(next_move)
            new_score, move = self.alphabeta_min(new_node, depth-1, alpha, beta)
            if new_score > max_score:
                max_score = new_score
                direction = next_move
            if max_score >= beta:
                return max_score, direction
            else:
                if alpha < max_score:
                    alpha = max_score

        return max_score, direction

    def alphabeta_min(self, game, depth, alpha, beta):
        # Get minimum score in Alpha-Beta pruning algorithm.

        if self.terminal_state(game, depth):
            return self.score(game, self), game.get_player_location(self)

        # Do not forget about time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        min_score = float("inf")
        direction = (-1, -1)

        for next_move in game.get_legal_moves():
            new_node = game.forecast_move(next_move)
            new_score, move = self.alphabeta_max(new_node, depth-1, alpha, beta)
            if new_score < min_score:
                direction = next_move
                min_score = new_score
            if min_score <= alpha:
                return min_score, direction
            else:
                if beta > min_score:
                    beta = min_score

        return min_score, direction
