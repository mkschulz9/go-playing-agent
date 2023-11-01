import math, random
from .node import _Node
from .mcts_config import get_parameters
from .utils.generate_boards import generate_valid_boards
from .utils.helper_functions import (player_won, position_played,
                             estimate_remaining_moves, calculate_ratios,
                             get_game_stage, get_player_scores)
from .heuristics.evaluate_move import (capture_score, liberty_score,
                                     blocking_score, territorial_gain,
                                     corner_edge_score, influence_score,
                                     atari_score, synergy_score)
from .heuristics.evaluate_board import (defensive_structures, count_potential_suicides,
                                      count_captured_spaces, count_total_eyes,
                                      count_alive_groups)

class MonteCarloTreeSearch:
    def __init__(self):
        self._root_node = None
        self.settings = get_parameters()
    
    # sets up root node of the MCTS tree
    # input: player number, current board layout, previous board layout
    # output: none
    def init_tree(self, player, current_board, previous_board):
        self._root_node = _Node(player, current_board, previous_board)
        
    # selects a leaf node to expand
    # input: None
    # output: leaf node selected for expansion
    def selection(self):
        current_node = self._root_node
        
        while current_node.children != [] and not current_node.simulated:
            log_parent_visits = math.log(current_node.simulation_visits) if current_node.simulation_visits else 0
            current_node = max(current_node.children, key=lambda child: self._modified_UCT(child, log_parent_visits))
            
        return current_node
    
    # decides to expand leaf node or return it directly
    # input: leaf node chosen during selection
    # output: node to start simulation from
    def expansion(self, leaf_node):
        if leaf_node.simulation_visits == 0:
            return leaf_node
        
        leaf_node.simulated = False
        leaf_node.children = []
        next_player = 3 - leaf_node.player
        
        possible_next_boards = generate_valid_boards(leaf_node.player, leaf_node.board, leaf_node.previous_board)
        possible_next_boards_move_scores = [self._evaluate_move(leaf_node.board, board, leaf_node.player) for board in possible_next_boards]

        sorted_pairs = sorted(zip(possible_next_boards_move_scores, possible_next_boards), reverse=True)
        sorted_boards = [board for _, board in sorted_pairs]
        sorted_scores = [score for score, _ in sorted_pairs]

        top_half_length = len(sorted_boards) // 2
        top_half_boards = sorted_boards[:max(1, top_half_length)]
        top_half_scores = sorted_scores[:max(1, top_half_length)]

        for i, board in enumerate(top_half_boards):
            child_node = _Node(next_player, board, leaf_node.board, leaf_node)
            child_node.move_score = top_half_scores[i]
            child_node.board_score = self._evaluate_board(board, self._root_node.player)
            leaf_node.children.append(child_node)
        
        if leaf_node.children == []:
            return leaf_node
        
        return leaf_node.children[0]
     
    # simulates a game from the given node either randomly or by choosing the 'best' move
    # input: node to start simulation from
    # output: node at the end of the simulation
    def simulation(self, starting_node):
        current_node = starting_node
        starting_node.simulated = True
        depth = 0
        
        estimated_moves_remaining = estimate_remaining_moves(current_node.board)
        valid_boards = generate_valid_boards(current_node.player, current_node.board, current_node.previous_board)
        
        if estimated_moves_remaining == 0 or not valid_boards:
            return current_node
        else:
            if random.random() < 0.7 and self._root_node.children:
                return current_node
            else:
                while True:
                    if not valid_boards or depth == estimated_moves_remaining:
                        return current_node
                    
                    if random.random() < 0.7:
                        best_board = max(valid_boards, key=lambda board: self._evaluate_move(current_node.board, board, current_node.player))
                        next_board = best_board
                    else:
                        next_board = random.choice(valid_boards)
                        
                    next_player = 3 - current_node.player
                    new_child = _Node(next_player, next_board, current_node.board, current_node)
                    current_node.children.append(new_child)
                    current_node = new_child
                    valid_boards = generate_valid_boards(current_node.player, current_node.board, current_node.previous_board)
                    depth += 1

    # backpropagates result of simulation, updating values in each node on path
    # input: node at end of simulation
    # output: none (modifies nodes attributes in winning path)
    def backpropagation(self, ending_node):
        current_node = ending_node
        value = self._evaluate_board(ending_node.board, self._root_node.player)
        winning_player = player_won(current_node.board)
        
        while current_node is not None:
            current_node.simulation_visits += 1
            current_node.total_value += value
            
            if current_node.parent_node != None:
                if current_node.parent_node.player == winning_player:
                    current_node.winning_simulation_visits += 1
            else:
                break
            
            current_node = current_node.parent_node
    
    # calculates UCT value for a given node
    # input: node to calculate UCT value for, logarithm of parent visits
    # output: UCT value of node
    def _modified_UCT(self, node, log_parent_visits):
        n_i = node.simulation_visits
        if n_i == 0:
            return float('inf')
        
        w_i = node.winning_simulation_visits
        game_stage = get_game_stage(node.board)
        stage_weights = self.settings['weights_UCT'][game_stage]
        
        uct_value = (stage_weights['alpha'] * ((w_i / n_i) * 100 if w_i != 0 and n_i != 0 else 0)) + \
                    (stage_weights['beta'] * node.move_score) + \
                    (stage_weights['gamma'] * math.sqrt(log_parent_visits / n_i if log_parent_visits != 0 and n_i != 0 else 0)) + \
                    (stage_weights['delta'] * node.board_score)
                    
        return uct_value
    
    # evaluates the quality of a move for a player
    # input: current board layout, next board layout, player
    # output: score of move for player
    def _evaluate_move(self, current_board, next_board, player):
        game_stage = get_game_stage(current_board)
        row, col = position_played(player, next_board, current_board)
        stage_weights = self.settings['weights_move'][game_stage]
        
        raw_scores = {
            'capture': capture_score(current_board, next_board, player),
            'liberty': liberty_score(current_board, next_board, player),
            'blocking': blocking_score(current_board, next_board, player),
            'territory': territorial_gain(current_board, next_board, player),
            'corner_edge': corner_edge_score(row, col),
            'influence': influence_score(next_board, row, col),
            'atari': atari_score(next_board, row, col, player),
            'synergy': synergy_score(next_board, row, col, player)
        }
        
        normalized_scores = {
            key: (raw_scores[key] - self.settings['min_max_values_move'][key][0]) / (self.settings['min_max_values_move'][key][1] - self.settings['min_max_values_move'][key][0])
            for key in raw_scores
        }

        return sum(normalized_scores[key] * stage_weights[key] for key in normalized_scores)
    
    # scores the given board
    # input: board, player
    # output: score of board for the player
    def _evaluate_board(self, board, player):
        game_stage = get_game_stage(board)
        player1_score, player2_score = get_player_scores(board)
        stage_weights = self.settings['weights_board'][game_stage]
        
        raw_scores = {
            'player_score': (player1_score - player2_score) if player == 1 else (player2_score - player1_score),
            'defensive': defensive_structures(board, player),
            'suicides': count_potential_suicides(board, 3 - player),
            'captured_spaces': count_captured_spaces(board, player),
            'eyes': count_total_eyes(board, player),
            'alive_groups': count_alive_groups(board, player)
        }

        normalized_scores = {
            key: (raw_scores[key] - self.settings['min_max_values_board'][key][0]) / (self.settings['min_max_values_board'][key][1] - self.settings['min_max_values_board'][key][0])
            for key in raw_scores
        }

        return sum(normalized_scores[key] * stage_weights[key] for key in normalized_scores)
    
    # updates player one weights to play more aggressively
    # input: None
    # output: None (modifies weights in-place)
    def update_player_1_weights(self):
        for index in range(1, len(self.settings['weights_move']) + 1):
            for key, multiplier in self.settings['move_multipliers'].items():
                self.settings['weights_move'][index][key] *= multiplier

        for index in range(1, len(self.settings['weights_board']) + 1):
            for key, multiplier in self.settings['board_multipliers'].items():
                self.settings['weights_board'][index][key] *= multiplier
    
    # calculates a root child's ending value
    # input: child node, game stage
    # output: value of child node
    def calc_child_value(self, child, game_stage):
        alpha, beta = self.settings['weights_final_move'][game_stage]
        win_sim_ratio, value_sim_ratio = calculate_ratios(alpha, beta, child)
        
        return win_sim_ratio + value_sim_ratio