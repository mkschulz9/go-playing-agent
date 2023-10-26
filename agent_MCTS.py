import math, random, time
from collections import deque

class MonteCarloTreeSearch:
    class _Node:
        def __init__(self, player, board, previous_board=None, parent_node=None, simulated = False):
            self.player = player
            self.board = board
            self.previous_board = previous_board
            self.parent_node = parent_node
            self.children = []
            self.move_score = 0
            self.board_score = 0
            self.total_value = 0
            self.simulated = simulated
            self.simulation_visits = 0
            self.winning_simulation_visits = 0
            #self.cached_uct = None
            
    def __init__(self, input_file_path):
        self._input_file_path = input_file_path
        self._root_node = None
        
    # selects a leaf node to expand
    # input: None
    # output: leaf node
    def selection(self):
        current_node = self._root_node
        
        while current_node.children != [] and not current_node.simulated:
            log_parent_visits = math.log(current_node.simulation_visits) if current_node.simulation_visits else 0
            current_node = max(current_node.children, key=lambda child: self._modified_UCT(child, log_parent_visits))
            
        return current_node
    
    # expands leaf node by adding all possible next moves as children under conditions
    # input: leaf node chosen during selection
    # output: node to start simulation from
    def expansion(self, leaf_node):
        if leaf_node.simulation_visits == 0:
            return leaf_node
        
        leaf_node.simulated = False
        leaf_node.children = []
        
        possible_next_boards = self._generate_valid_boards(leaf_node.player, leaf_node.board, leaf_node.previous_board)
        possible_next_boards_move_scores = [self._evaluate_move(leaf_node.board, board, leaf_node.player) for board in possible_next_boards]

        sorted_pairs = sorted(zip(possible_next_boards_move_scores, possible_next_boards), reverse=True)

        sorted_boards = [board for _, board in sorted_pairs]
        sorted_scores = [score for score, _ in sorted_pairs]

        top_half_length = len(sorted_boards) // 2
        top_half_boards = sorted_boards[:max(1, top_half_length)]
        top_half_scores = sorted_scores[:max(1, top_half_length)]

        next_player = 3 - leaf_node.player

        for i, board in enumerate(top_half_boards):
            child_node = self._Node(next_player, board, leaf_node.board, leaf_node)
            child_node.move_score = top_half_scores[i]
            child_node.board_score = self._evaluate_board(board, self._root_node.player)
            leaf_node.children.append(child_node)
        
        if leaf_node.children == []:
            return leaf_node
        
        return leaf_node.children[0]
     
    # simulates game randomly from the given node
    # input: node to start simulation from
    # output: node at the end of the simulation
    def simulation(self, starting_node):
        current_node = starting_node
        starting_node.simulated = True
        depth = 0
        
        estimated_moves_remaining = self._estimate_remaining_moves(current_node.board)
        valid_boards = self._generate_valid_boards(current_node.player, current_node.board, current_node.previous_board)
        
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
                    new_child = self._Node(next_player, next_board, current_node.board, current_node)
                    current_node.children.append(new_child)
                    current_node = new_child
                    valid_boards = self._generate_valid_boards(current_node.player, current_node.board, current_node.previous_board)
                    depth += 1

    # backpropagates result of simulation, updating each node on path
    # input: node at end of simulation, boolean indicating if root player won
    # output: none (modifies nodes attributes in winning path)
    def backpropagation(self, ending_node):
        value = self._evaluate_board(ending_node.board, self._root_node.player)
        current_node = ending_node
        winning_player = self._player_won(current_node.board)
        
        while current_node is not None:
            current_node.simulation_visits += 1
            current_node.total_value += value
            
            if current_node.parent_node != None:
                if current_node.parent_node.player == winning_player:
                    current_node.winning_simulation_visits += 1
            else:
                break
            
            current_node = current_node.parent_node
    
    
    # sets up root node of the MCTS tree
    # input: player number, current board layout, previous board layout
    # Output none
    def init_tree(self, player, current_board, previous_board):
        self._root_node = self._Node(player, current_board, previous_board)

    # reads & parses input file
    # input: None
    # output: color number agent is playing as, board layout after agent's last move, board layout after opponent's last move
    def read_input(self):
        with open(self._input_file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]

        player = int(lines[0])
        previous_board = [list(map(int, list(line))) for line in lines[1:6]]
        current_board = [list(map(int, list(line))) for line in lines[6:11]]

        return player, current_board, previous_board
    
    # writes output to file on first file in format: 'row,column'
    # input: position played by agent
    # output: none (writes to file)
    def write_output(self, move_played):
        with open('output.txt', 'w') as file:
            file.write(f"{move_played[0]},{move_played[1]}")
            
    def get_opening_move(self, player, current_board):
        if player == 1:
            return (2, 2)
        else:
            if current_board[2][2] == 0:
                return (2, 2)
            else:
                edge_positions = [(0, i) for i in range(5)] + \
                                [(4, i) for i in range(5)] + \
                                [(i, 0) for i in range(1, 4)] + \
                                [(i, 4) for i in range(1, 4)]

                random.shuffle(edge_positions)

                for x, y in edge_positions:
                    if current_board[x][y] == 0:
                        return (x, y)
                
    
    '''
    # checks if the value is stable and the confidence interval is narrow enough to stop early
    # input: previous highest UCT value, epsilon, number of stable iterations, number of iterations to stop after, confidence interval
    # output: boolean indicating if the algorithm should stop early, highest UCT value, number of stable iterations
    def check_early_stop(self, prev_highest_uct, epsilon, num_stable_iterations, stopping_iterations, confidence_interval):
        log_parent_visits = math.log(self._root_node.simulation_visits) if self._root_node.simulation_visits else 0
        child_node_highest_uct, highest_uct = max(
            ((child, self._modified_UCT(child, log_parent_visits, True)) for child in self._root_node.children),
            key=lambda x: x[1]
        )        
        
        if abs(highest_uct - prev_highest_uct) < epsilon:
            num_stable_iterations += 1
            if num_stable_iterations >= stopping_iterations:
                if child_node_highest_uct.simulation_visits > 0:
                    inside_sqrt = (highest_uct * (1 - highest_uct)) / child_node_highest_uct.simulation_visits
                    if inside_sqrt >= 0:
                        confidence = 1.96 * math.sqrt(inside_sqrt)
                        if confidence < confidence_interval:
                            return True, 0, 0
        else:
            num_stable_iterations = 0
            
        return False, highest_uct, num_stable_iterations
    '''
     
    # generates all valid boards that can be reached from the given board by the given player
    # input: player, current board layout, previous board layout
    # output: valid boards that can be reached
    def _generate_valid_boards(self, player, current_board, previous_board):
        valid_boards = []
        captured_stones = []

        positions = [(row, col) for row in range(5) for col in range(5)]
        for row, col in positions:
            if current_board[row][col] == 0:
                self._place_stone_capture(player, current_board, row, col, captured_stones)
                
                if not self._count_liberties(player, current_board, row, col) == 0 and not current_board == previous_board:
                    valid_boards.append([row.copy() for row in current_board])
                
                self._undo_move(player, current_board, row, col, captured_stones)
                captured_stones.clear()

        return valid_boards


    # places a stone for the given player and captures any opponent stones if they are left without liberties
    # input: player, board to place stone on, row & column to place stone, list to keep track of captured stones
    # output: none (modifies the board variable in-place)
    def _place_stone_capture(self, player, board, row, col, captured_stones):
        board[row][col] = player
        opponent = 3 - player

        for neighbor_row, neighbor_col in self._get_neighbors(row, col):
            if board[neighbor_row][neighbor_col] == opponent and self._count_liberties(opponent, board, neighbor_row, neighbor_col) == 0:
                self._capture_group(opponent, board, neighbor_row, neighbor_col, captured_stones)
        
        return len(captured_stones)
    
    # undoes a move by the given player and restores the board to its original state
    # input: player, board to undo move on, row & column to undo move, list of captured stones to restore
    # output: none (modifies the board variable in-place)
    def _undo_move(self, player, board, row, col, captured_stones):
        board[row][col] = 0
        
        for r, c in captured_stones:
            board[r][c] = 3 - player
            
    # counts the number of liberties (empty adjacent cells) for the stone/group starting from the given cell
    # input: player, board layout, row & column of cell
    # output: number of liberties of the stone/group
    def _count_liberties(self, player, board, row, col):
        visited = set()
        to_check = [(row, col)]
        liberties = 0
        
        while to_check:
            current_row, current_col = to_check.pop()
            visited.add((current_row, current_col))
            
            for neighbor_row, neighbor_col in self._get_neighbors(current_row, current_col):
                if board[neighbor_row][neighbor_col] == 0:
                    liberties += 1
                elif board[neighbor_row][neighbor_col] == player and (neighbor_row, neighbor_col) not in visited:
                    to_check.append((neighbor_row, neighbor_col))
        
        return liberties

    # capture an entire group of opponent's stones starting from the given cell
    # input: player, board layout, row & column of cell, list to keep track of captured stones
    # output: none (modifies the board variable in-place)
    def _capture_group(self, player_to_capture, board, start_row, start_col, captured_stones):
        to_capture = [(start_row, start_col)]
        visited = set()

        while to_capture:
            current_row, current_col = to_capture.pop()
            board[current_row][current_col] = 0
            captured_stones.append((current_row, current_col))
            visited.add((current_row, current_col))

            for neighbor_row, neighbor_col in self._get_neighbors(current_row, current_col):
                if board[neighbor_row][neighbor_col] == player_to_capture and (neighbor_row, neighbor_col) not in visited:
                    to_capture.append((neighbor_row, neighbor_col))
                    
    # gets all valid neighboring cells for the given cell
    # input: row & column of cell
    # output: list of valid neighboring cells
    def _get_neighbors(self, row, col):
        neighbors = []
        if row > 0:
            neighbors.append((row-1, col))
        if row < 4:
            neighbors.append((row+1, col))
        if col > 0:
            neighbors.append((row, col-1))
        if col < 4:
            neighbors.append((row, col+1))
        return neighbors
    
    # calculates UCT value for a given node
    # input: node to calculate UCT value for, logarithm of parent visits
    # output: UCT value of node
    def _modified_UCT(self, node, log_parent_visits, checking_early_stop = False):
        n_i = node.simulation_visits
        w_i = node.winning_simulation_visits
        game_stage = self._get_game_stage(node.board)
        
        if n_i == 0:
            return float('inf')

        # Beg-game
        if game_stage == 1:
            alpha = 0.001  
            beta = 0.001   
            gamma = 1  
            delta = 0.001  
        # Mid-game
        elif game_stage == 2:
            alpha = 0.01  
            beta = 0.01   
            gamma = 0.6  
            delta = 0.1
        # End-game
        else:
            alpha = 0.1  
            beta = 0.1   
            gamma = 20  
            delta = 0.1
        
        uct_value = (alpha * ((w_i / n_i) * 100 if w_i != 0 and n_i != 0 else 0)) + \
                    (beta * node.move_score) + \
                    (gamma * math.sqrt(log_parent_visits / n_i if log_parent_visits != 0 and n_i != 0 else 0))
                    
        if not checking_early_stop:
            uct_value += (delta * node.board_score)
                    
        return uct_value
    
    def _get_game_stage(self, board):
        moves_remaining = self._estimate_remaining_moves(board)
        
        if moves_remaining > 16:
            return 1
        elif moves_remaining > 8:
            return 2
        else:
            return 3

    # evaluates the quality of a move for a player
    # input: current board layout, next board layout, player
    # output: score of move for player
    def _evaluate_move(self, current_board, next_board, player):
        weights = {
            1: {'capture': 1, 'liberty': 2, 'blocking': 2, 'territory': 1, 'corner_edge': 2, 'influence': 2, 'atari': 1, 'synergy': 1},
            2: {'capture': 1, 'liberty': 1, 'blocking': 2, 'territory': 2, 'corner_edge': 1, 'influence': 1, 'atari': 1, 'synergy': 2},
            3: {'capture': 1, 'liberty': 1, 'blocking': 1, 'territory': 2, 'corner_edge': 1, 'influence': 1, 'atari': 2, 'synergy': 2}
        }
        game_stage = self._get_game_stage(current_board)
        stage_weights = weights[game_stage]
        row, col = self._position_played(player, next_board, current_board)
        
        score = 0
        score += self._capture_score(current_board, next_board, player) * stage_weights['capture']
        score += self._liberty_score(current_board, next_board, player) * stage_weights['liberty']
        score += self._blocking_score(current_board, next_board, player) * stage_weights['blocking']
        score += self._territorial_gain(current_board, next_board, player) * stage_weights['territory']
        score += self._corner_edge_score(row, col) * stage_weights['corner_edge']
        score += self._influence_score(next_board, row, col) * stage_weights['influence']
        score += self._atari_score(next_board, row, col, player) * stage_weights['atari']
        score += self._synergy_score(next_board, row, col, player) * stage_weights['synergy']
        
        return score
    
    # scores the given board based on the number of stones captured by the player
    # input: current board layout, next board layout, player
    # output: score of board for player
    def _capture_score(self, current_board, next_board, player):
        captured_stones = 0
        for x in range(5):
            for y in range(5):
                if current_board[x][y] == 3 - player and next_board[x][y] == 0:
                    captured_stones += 1
        return captured_stones
    
    # calculates the total number of liberties for the given player
    # input: board layout, player
    # output: total number of liberties for player
    def _calculate_liberties(self, board, player):
        total_liberties = 0
        for row in range(5):
            for col in range(5):
                if board[row][col] == player:
                    total_liberties += self._count_liberties(player, board, row, col)
        return total_liberties

    # scores the boards based on the number of liberties gained by the player
    # input: current board layout, next board layout, player
    # output: difference in liberties gained by player
    def _liberty_score(self, current_board, next_board, player):
        current_liberties = self._calculate_liberties(current_board, player)
        next_liberties = self._calculate_liberties(next_board, player)
        return next_liberties - current_liberties
    
    # scores the boards based on the number of liberties lost by the opponent
    # input: current board layout, next board layout, player
    # output: difference in liberties lost by opponent
    def _blocking_score(self, current_board, next_board, player):
        opponent = 3 - player
        current_opponent_liberties = self._calculate_liberties(current_board, opponent)
        next_opponent_liberties = self._calculate_liberties(next_board, opponent)
        return current_opponent_liberties - next_opponent_liberties
    
    # determines how a move helps secure or extend the player's influence over certain areas of the board
    # input: current board layout, next board layout, player
    # output: difference in territory gained by player
    def _territorial_gain(self, current_board, next_board, player):
        opponent = 3 - player
    
        initial_territory = self._count_potential_territory(current_board, player)
        new_territory = self._count_potential_territory(next_board, player)
        
        initial_opponent_territory = self._count_potential_territory(current_board, opponent)
        new_opponent_territory = self._count_potential_territory(next_board, opponent)
        
        territorial_difference = (new_territory - initial_territory) - (new_opponent_territory - initial_opponent_territory)
        
        return territorial_difference
    
    # counts the number of potential territories for the given player
    # input: board layout, player
    # output: number of potential territories for player
    def _count_potential_territory(self, board, player):
        territory_count = 0
        for row in range(5):
            for col in range(5):
                if board[row][col] == 0:
                    neighbors = self._get_neighbors(row, col)
                    if all(board[n_row][n_col] == player for n_row, n_col in neighbors):
                        territory_count += 1
                    else:
                        break
        return territory_count
    
    # determines if a move is in a corner or along an edge
    # row and column of move placed
    # value of move based on edge or corner placement
    def _corner_edge_score(self, row, col):
        score = 0
        if row in [0, 4] and col in [0, 4]:
            score += 2
        elif row in [0, 4] or col in [0, 4]:
            score += 1
        return score
    
    # evaluates the overall 'influence' a stone could have on surrounding empty points
    # board, row, and column placed
    # value of stones 'influence'
    def _influence_score(self, board, row, col):
        score = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_row, new_col = row + dx, col + dy
                if 0 <= new_row < 5 and 0 <= new_col < 5 and board[new_row][new_col] == 0:
                    score += 1
        return score

    # evaluates if the move places the opponent into atari
    # board, row, column, and player
    # value of placing opponent into atari
    def _atari_score(self, board, row, col, player):
        score = 0
        opponent = 3 - player
        for neighbor_row, neighbor_col in self._get_neighbors(row, col):
            if board[neighbor_row][neighbor_col] == opponent and self._count_liberties(opponent, board, neighbor_row, neighbor_col) == 1:
                score += 3
        return score
    
    # evaluates the potential for future value of current placement
    # board, row, column, player
    # score of how well move sets up future play
    def _synergy_score(self, board, row, col, player):
        score = 0
        for neighbor_row, neighbor_col in self._get_neighbors(row, col):
            if board[neighbor_row][neighbor_col] == player:
                score += 2
        return score
        
    # estimates the number of moves remaining in the game
    # input: board layout
    # output: estimated number of moves remaining
    def _estimate_remaining_moves(self, board):
        total_moves = 24
        
        player1_stones, player2_stones  = self._get_player_scores(board)
        player2_stones -= 2.5
        
        if player1_stones == player2_stones:
            remaining_moves = total_moves - (player1_stones + player2_stones)
        elif player1_stones > player2_stones:
            remaining_moves = total_moves - (player1_stones * 2)
        else:
            remaining_moves = total_moves - (player2_stones * 2)
        
        return remaining_moves
    
    # scores the given terminal board for the root player based on the number of stones and eyes of root player
    # input: board, player
    # output: score of board for root player
    # POSSIBLY ADD A GAME PHASE PARAMETER (In the late game, the focus often shifts from capturing stones to solidifying territory. You could weight your evaluation metrics differently based on the game phase.)
    def _evaluate_board(self, board, player):
        weights = {
            1: {'player_score': 1, 'defensive': 1, 'suicides': 1, 'captured_spaces': 1, 'eyes': 1, 'alive_groups': 1},
            2: {'player_score': 1.5, 'defensive': 2, 'suicides': 1, 'captured_spaces': 2, 'eyes': 1, 'alive_groups': 1},
            3: {'player_score': 2, 'defensive': 1, 'suicides': 1, 'captured_spaces': 2, 'eyes': 1, 'alive_groups': 3}
        }
        game_stage = self._get_game_stage(board)
        stage_weights = weights[game_stage]
        score = 0
        player1_score, player2_score = self._get_player_scores(board)
        opponent = 3 - player
                
        if player == 1:
            score += (player1_score - player2_score) * stage_weights['player_score']
        else:
            score += (player2_score - player1_score) * stage_weights['player_score']

        score += self._defensive_structures(board, player) * stage_weights['defensive']
        score += self._count_potential_suicides(board, opponent) * stage_weights['suicides']
        score += self._count_captured_spaces(board, player) * stage_weights['captured_spaces']
        score += self._count_total_eyes(board, player) * stage_weights['eyes']
        score += self._count_alive_groups(board, player) * stage_weights['alive_groups']
        
        return score
    
    # counts the number of spaces captured by the given player
    # input: board layout, player
    # output: number of spaces captured by player
    def _count_captured_spaces(self, board, player):
        captured_spaces = 0
        visited = set()

        for row in range(5):
            for col in range(5):
                if board[row][col] == 0 and (row, col) not in visited:
                    is_captured, space_count, space_visited = self._is_space_captured(board, row, col, player)
                    if is_captured:
                        captured_spaces += space_count
                    visited.update(space_visited)

        return captured_spaces

    # determines if the given space is captured by the given player
    # input: board layout, row & column of space, player
    # output: boolean indicating if space is captured, number of spaces captured, set of visited spaces
    def _is_space_captured(self, board, start_row, start_col, player):
        to_check = [(start_row, start_col)]
        visited = set()
        is_captured = True
        space_count = 0

        while to_check:
            current_row, current_col = to_check.pop()
            visited.add((current_row, current_col))
            space_count += 1

            for neighbor_row, neighbor_col in self._get_neighbors(current_row, current_col):
                if board[neighbor_row][neighbor_col] == 0 and (neighbor_row, neighbor_col) not in visited:
                    to_check.append((neighbor_row, neighbor_col))
                elif board[neighbor_row][neighbor_col] != player:
                    is_captured = False

        return is_captured, space_count, visited
    
    # determines if the number of stones for each player is within 2 of each other
    # input: board layout
    # output: boolean indicating if the number of stones for each player is within 2 of each other
    def _check_equal_pieces(self, board):
        count_player1 = 0
        count_player2 = 0

        for row in board:
            for cell in row:
                if cell == 1:
                    count_player1 += 1
                elif cell == 2:
                    count_player2 += 1

        return abs(count_player1 - count_player2) <= 2

    # counts the number of groups of stones with high liberties for the given player
    # input: board layout, player
    # output: number of groups of stones with high liberties for the given player
    def _defensive_structures(self, board, player):
        high_liberty_count = 0
        visited = set()

        for row in range(5):
            for col in range(5):
                if (row, col) not in visited and board[row][col] == player:
                    group_liberties = self._count_liberties(player, board, row, col)
                    
                    if group_liberties >= 2:
                        high_liberty_count += 1

                    to_check = [(row, col)]
                    while to_check:
                        current_row, current_col = to_check.pop()
                        visited.add((current_row, current_col))
                        
                        for neighbor_row, neighbor_col in self._get_neighbors(current_row, current_col):
                            if board[neighbor_row][neighbor_col] == player and (neighbor_row, neighbor_col) not in visited:
                                to_check.append((neighbor_row, neighbor_col))

        return high_liberty_count
    
    def _deep_territory_score(self, board, player):
        score = 0
        for row in range(5):
            for col in range(5):
                if board[row][col] == 0:
                    depth = self._calculate_depth_from_player_stones((row, col), board, player)
                    score += depth
        return score

    def _calculate_depth_from_player_stones(point, board, player):
        rows, cols = len(board), len(board[0])
        start_row, start_col = point
        visited = set()
        queue = deque([(start_row, start_col, 0)])

        while queue:
            curr_row, curr_col, depth = queue.popleft()
            
            if (curr_row, curr_col) in visited:
                continue
            visited.add((curr_row, curr_col))

            if board[curr_row][curr_col] == 3 - player:
                return depth

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = curr_row + dr, curr_col + dc

                if 0 <= new_row < rows and 0 <= new_col < cols:
                    queue.append((new_row, new_col, depth + 1))

        return depth
    
    def _count_eyes(self, board, group):
        eye_count = 0
        for row, col in group:
            neighbors = self._get_neighbors(row, col)
            if all(board[r][c] == board[row][col] for r, c in neighbors):
                eye_count += 1
        return eye_count
    
    def _count_total_eyes(self, board, player):
        eye_count = 0

        for row in range(5):
            for col in range(5):
                if board[row][col] == 0:
                    neighbors = self._get_neighbors(row, col)
                    if all(board[r][c] == player for r, c in neighbors):
                        eye_count += 1

        return eye_count
    
    # counts the number of potential suicides for the root player's opponent
    # input: board layout, opponent player number
    # output: number of potential suicides for opponent
    def _count_potential_suicides(self, board, opponent):
        potential_suicides = 0

        for row in range(5):
            for col in range(5):
                if board[row][col] == 0:
                    neighbors = self._get_neighbors(row, col)
                    
                    if all(board[r][c] != opponent for r, c in neighbors):
                        potential_suicides += 1

        return potential_suicides
    
    # counts number of groups with at least two eyes
    def _count_alive_groups(self, board, player):
        alive_groups = 0
        visited = set()
        for row in range(5):
            for col in range(5):
                if (row, col) not in visited and board[row][col] == player:
                    group = []  # Stores the stones that belong to this group
                    to_check = [(row, col)]
                    while to_check:
                        current_row, current_col = to_check.pop()
                        visited.add((current_row, current_col))
                        group.append((current_row, current_col))
                        
                        for neighbor_row, neighbor_col in self._get_neighbors(current_row, current_col):
                            if board[neighbor_row][neighbor_col] == player and (neighbor_row, neighbor_col) not in visited:
                                to_check.append((neighbor_row, neighbor_col))
                    
                    if self._count_eyes(board, group) >= 2:
                        alive_groups += 1
        return alive_groups
    
    # calculates scores of each opponent
    # board
    # scores for each player
    def _get_player_scores(self, board):
        komi_value = 2.5
        player1_score = 0
        player2_score = 0

        for row in board:
            for cell in row:
                if cell == 1:
                    player1_score += 1
                elif cell == 2:
                    player2_score += 1

        player2_score += komi_value
        
        return player1_score, player2_score
    
    # determines if the root player has won the game
    # input: board layout at end of game
    # output: True if root player has won, False otherwise
    def _player_won(self, board):
        player1_score, player2_score = self._get_player_scores(board)

        return 1 if player1_score > player2_score else 2
    
    # returns the position played by the player
    # input: player, current board layout, previous board layout
    # output: position played by the player    
    def _position_played(self, player, current_board, previous_board):
        for row in range(5):
            for col in range(5):
                if current_board[row][col] == player and previous_board[row][col] != player:
                    return (row, col)
        return None

def main():
    start_time = time.time()
    #num_stable_iterations = 0
    #stopping_iterations = 40
    max_time = 7
    #epsilon = 0.01
    #confidence_interval = 0.05
    #prev_highest_uct = float('-inf')
    
    agent_MCTS = MonteCarloTreeSearch("./input.txt")
    player, current_board, previous_board = agent_MCTS.read_input()
    agent_MCTS.init_tree(player, current_board, previous_board)
    
    # check if we are starting the game
    number_pieces = 0
    for row in current_board:
        for cell in row:
            if cell != 0:
                number_pieces += 1
    
    if number_pieces < 2:
        move_played = agent_MCTS.get_opening_move(player, current_board)
        agent_MCTS.write_output(move_played)
        exit()
        
    # check if board is empty
    
    while time.time() - start_time < max_time:
        #num_iterations += 1
        leaf_node = agent_MCTS.selection()        
        child_node = agent_MCTS.expansion(leaf_node)        
        ending_node = agent_MCTS.simulation(child_node)
        agent_MCTS.backpropagation(ending_node)
        
        '''
        stop_early, highest_uct, num_stable_iterations = agent_MCTS.check_early_stop(prev_highest_uct, epsilon, num_stable_iterations, stopping_iterations, confidence_interval)
        if stop_early:
            print("Stopping Early")
            break
        else:
            prev_highest_uct = highest_uct
        '''
        
        game_stage = agent_MCTS._get_game_stage(current_board)

        # Beg-game
        if game_stage == 1:
            alpha = 0.001  
            beta = 0.1   
            gamma = 0.01   
        # Mid-game
        elif game_stage == 2:
            alpha = 0.5  
            beta = 0.05   
            gamma = 0.1  
        # End-game
        else:
            alpha = 0.5  
            beta = 0.01   
            gamma = 0.5  
            
    child = max(agent_MCTS._root_node.children, key=lambda child: (alpha * ((child.winning_simulation_visits / child.simulation_visits) * 100 if child.winning_simulation_visits != 0 and child.simulation_visits != 0 else 0)) + (beta * child.move_score) + (gamma * (child.total_value / child.simulation_visits if child.total_value != 0 and child.simulation_visits != 0 else 0)))
    
    '''
    # print the 'value' of each child of root
    for x in agent_MCTS._root_node.children: 
        print(f"Child Number: {agent_MCTS._root_node.children.index(x)}")
        print(f"Winning Visits: {x.winning_simulation_visits}, Visits: {x.simulation_visits}, Final Score: {(alpha * ((x.winning_simulation_visits / x.simulation_visits) * 100 if x.winning_simulation_visits != 0 and x.simulation_visits != 0 else 0)) + (beta * child.move_score) + (gamma * (x.total_value / x.simulation_visits if x.total_value != 0 and x.simulation_visits != 0 else 0))}\n")
    print(f"Child Chosen Number: {agent_MCTS._root_node.children.index(child)}")
    # print information about the distribution of the search
    # print total number of simulations
    # print percent of simulations for each child, print in format xx.xx%
    for x in agent_MCTS._root_node.children:
        print(f"Child Number: {agent_MCTS._root_node.children.index(x)}, Percent of Simulations: {round((x.simulation_visits / agent_MCTS._root_node.simulation_visits) * 100, 2)}%")
    '''
    
    move_played = agent_MCTS._position_played(player, child.board, current_board)
    agent_MCTS.write_output(move_played)
    
if __name__ == "__main__":
    main()
    
# Notes:
# - best known moves for beginning and ending game
# - handle if input has 'PASS'
# - playing as black -> have to capture to win
# - consider passing at the very end of game