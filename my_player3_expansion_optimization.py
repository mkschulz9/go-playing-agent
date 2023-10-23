import time, math, random

class MonteCarloTreeSearch:
    class _Node:
        def __init__(self, player, board, previous_board=None, parent_node=None, simulated = False):
            self.player = player
            self.board = board
            self.previous_board = previous_board
            self.parent_node = parent_node
            self.parent_simulation_visits = None
            self.children = []
            self.simulated = simulated
            self.simulation_visits = 0
            self.winning_simulation_visits = 0
            self.cached_uct = None
            
    def __init__(self, input_file_path):
        self._input_file_path = input_file_path
        self._root_node = None
        
    # selects a leaf node to expand
    # input: None
    # output: leaf node
    def selection(self):
        current_node = self._root_node
        
        while current_node.children != [] and not current_node.simulated:
            log_parent_visits = math.log(current_node.simulation_visits) if current_node.simulation_visits else 1.0
            current_node = max(current_node.children, key=lambda child: self._UCT(child, log_parent_visits))
    
        return current_node
    
    # expands leaf node by adding all possible next moves as children under conditions
    # input: leaf node chosen during selection
    # output: node to start simulation from
    def expansion(self, leaf_node):
        if leaf_node.simulation_visits == 0:
            return leaf_node
        
        leaf_node.simulated = False
        
        possible_next_boards = self._generate_valid_boards(leaf_node.player, leaf_node.board, leaf_node.previous_board)
        promising_boards = [board for board in possible_next_boards if self._is_promising_move(leaf_node.player, leaf_node.board, board)]
        evaluated_boards = [(board, *self._board_evaluation(board, leaf_node.player)) for board in promising_boards]
        promising_boards = [(board, score) for board, score, is_promising in evaluated_boards if is_promising]
        sorted_promising_boards = sorted(promising_boards, key=lambda x: x[1])
        boards_only = [board for board, _ in sorted_promising_boards]
        
        if boards_only == []:
            evaluated_boards = [(board, *self._board_evaluation(board, leaf_node.player)) for board in possible_next_boards]
            sorted_evaluated_boards = sorted(evaluated_boards, key=lambda x: x[1], reverse=True)
            top_n = len(evaluated_boards) // 2
            boards_only = [board for board, _, _ in sorted_evaluated_boards[:top_n]]
        
        
        for board in boards_only:
            child_node = self._Node(self._opponent_player(leaf_node.player), board, leaf_node.board, leaf_node)
            leaf_node.children.append(child_node)
        
        if leaf_node.children == []:
            return leaf_node
        
        return random.choice(leaf_node.children)
     
    # simulates game randomly from the given node
    # input: node to start simulation from
    # output: node at the end of the simulation
    def simulation(self, starting_node):
        current_node = starting_node
        starting_node.simulated = True
        
        while True:
            valid_boards = self._generate_valid_boards(current_node.player, current_node.board, current_node.previous_board)
            if valid_boards == []:
                return current_node, self._has_root_player_won(current_node.board)
                
            new_child = self._Node(self._opponent_player(current_node.player), random.choice(valid_boards), current_node.board, current_node)
            current_node.children.append(new_child)
            current_node = new_child

    # backpropagates result of simulation, updating each node on path
    # input: node at end of simulation, boolean indicating if root player won
    # output: none (modifies nodes attributes in winning path)
    def backpropagation(self, ending_node, root_player_won):
        current_node = ending_node
        
        if root_player_won:
            player_won = self._root_node.player
        else:
            player_won = self._opponent_player(self._root_node.player)

        while current_node.parent_node:
            current_node.cached_uct = None
            current_node.simulation_visits += 1
            if current_node.player != player_won:
                current_node.winning_simulation_visits += 1
            current_node = current_node.parent_node
            
        # sets up root node and its children of the MCTS tree
    # input: player number, current board layout, previous board layout
    # Output none
    def init_tree(self, player, current_board, previous_board):
        self._root_node = self._Node(player, current_board, previous_board)
        possible_next_boards = self._generate_valid_boards(player, current_board, previous_board)
        next_player = 3 - player
        
        for board in possible_next_boards:
            child_node = self._Node(next_player, board, self._root_node.board, self._root_node)
            self._root_node.children.append(child_node)
        
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
            
    # calculates UCT value for a given node
    # input: node to calculate UCT value for, logarithm of parent visits
    # output: UCT value of node
    def _UCT(self, node, log_parent_visits):
        
        if node.cached_uct is not None and node.parent_simulation_visits == node.parent_node.simulation_visits:
            return node.cached_uct
        
        C = math.sqrt(2)
        n_i = node.simulation_visits
        w_i = node.winning_simulation_visits
                
        if n_i == 0:
            return float('inf')
        
        uct_value = (w_i / n_i) + C * math.sqrt(log_parent_visits / n_i)
        node.cached_uct = uct_value
        
        node.cached_uct = uct_value
        node.parent_simulation_visits = node.parent_node.simulation_visits
        return uct_value
    
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
    
    
    # evaluates the given board for the given player
    # input: board layout, player
    # output: score of board for player, boolean indicating if board is promising
    def _board_evaluation(self, board, player):
        is_promising = False
        score = 0
        potential_opponent_suicides = 0
        root_player = self._root_node.player
        opponent = self._opponent_player(player)
        
        root_player_stones = sum(row.count(root_player) for row in board)     
        opponent_stones = sum(row.count(opponent) for row in board)   
        groups_with_high_liberties = self._defensive_structures(board, root_player)
        if self._check_equal_pieces(board):
            potential_opponent_suicides = self._count_potential_suicides(board, opponent)
        captured_spaces = self._count_captured_spaces(board, root_player)
        
        score += root_player_stones * 10
        score += groups_with_high_liberties
        score += potential_opponent_suicides
        score += captured_spaces
        
        if root_player_stones > opponent_stones and groups_with_high_liberties > 0 and captured_spaces > 0:
            is_promising = True
        
        return score, is_promising
    
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

    # determines if the given move is promising for the player
    # input: player, current board layout, next board layout
    # output: boolean indicating if move is promising
    def _is_promising_move(self, player, current_board, next_board):
     
        capture_score = self._capture_score(current_board, next_board, player)
        liberty_score = self._liberty_score(current_board, next_board, player)
        blocking_score = self._blocking_score(current_board, next_board, player)
        territorial_gain = self._territorial_gain(current_board, next_board, player) 
        
        # consider changing this to take into account the averages for each metric & favor metrics at different points in the game
        if capture_score > 0 and liberty_score > 0 and blocking_score > 0 and territorial_gain > 0 or capture_score >= 1:
            return True
        else:
            return False
    
    # scores the given board based on the number of stones captured by the player
    # input: current board layout, next board layout, player
    # output: score of board for player
    def _capture_score(self, current_board, next_board, player):
        captured_stones = 0
        opponent = self._opponent_player(player)
        
        for x in range(5):
            for y in range(5):
                if current_board[x][y] == opponent and next_board[x][y] == 0:
                    captured_stones += 1
                    
        return captured_stones

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
        opponent = self._opponent_player(player)
        current_opponent_liberties = self._calculate_liberties(current_board, opponent)
        next_opponent_liberties = self._calculate_liberties(next_board, opponent)
        return current_opponent_liberties - next_opponent_liberties
    
    # determines how a move helps secure or extend the player's influence over certain areas of the board
    # input: current board layout, next board layout, player
    # output: difference in territory gained by player
    def _territorial_gain(self, current_board, next_board, player):
        opponent = self._opponent_player(player)
    
        initial_territory = self._count_potential_territory(current_board, player)
        new_territory = self._count_potential_territory(next_board, player)
        
        initial_opponent_territory = self._count_potential_territory(current_board, opponent)
        new_opponent_territory = self._count_potential_territory(next_board, opponent)
        
        territorial_difference = (new_territory - initial_territory) - (new_opponent_territory - initial_opponent_territory)
        return territorial_difference

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
    
    # copies the given board
    # input: board to copy
    # output: copy of board
    def _copy_board(self, board):
        return [row[:] for row in board]
        
    # returns the opponent of the given player
    # input: player
    # output: opponent of player
    def _opponent_player(self, player):
        return 3 - player
    
    # checks if the value is stable and the confidence interval is narrow enough to stop early
    # input: previous highest UCT value, epsilon, number of stable iterations, number of iterations to stop after, confidence interval
    # output: boolean indicating if the algorithm should stop early, highest UCT value, number of stable iterations
    def _check_early_stop(self, prev_highest_uct, epsilon, num_stable_iterations, stopping_iterations, confidence_interval):
        log_parent_visits = math.log(self._root_node.simulation_visits) if self._root_node.simulation_visits else 1.0
        child_node_highest_uct, highest_uct = max(
            ((child, self._UCT(child, log_parent_visits)) for child in self._root_node.children),
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
        
    # returns the position played by the player
    # input: player, current board layout, previous board layout
    # output: position played by the player    
    def _position_played(self, player, current_board, previous_board):
        for row in range(5):
            for col in range(5):
                if current_board[row][col] == player and previous_board[row][col] != player:
                    return (row, col)
        return None

    # determines if the root player has won the game
    # input: board layout at end of game
    # output: True if root player has won, False otherwise
    def _has_root_player_won(self, ending_board):
        komi_value = 2.5

        player1_score = sum(row.count(1) for row in ending_board)
        player2_score = sum(row.count(2) for row in ending_board) + komi_value
        
        if self._root_node.player == 1:
            root_score = player1_score
            opponent_score = player2_score
        else:
            root_score = player2_score
            opponent_score = player1_score
        
        return root_score > opponent_score

# main function, runs the MCTS algorithm and writes output to file
def main():
    start_time = time.time()
    prev_highest_uct = float('-inf')
    epsilon = 0.01
    num_stable_iterations = 0
    stopping_iterations = 50
    confidence_interval = 0.05
    max_time = 7.5
    
    
    agent_MCTS = MonteCarloTreeSearch("./input.txt")
    player, current_board, previous_board = agent_MCTS.read_input()
    agent_MCTS.init_tree(player, current_board, previous_board)
    
    while time.time() - start_time < max_time:
        leaf_node = agent_MCTS.selection()   
        child_node = agent_MCTS.expansion(leaf_node)   
        ending_node, root_player_won = agent_MCTS.simulation(child_node)
        agent_MCTS.backpropagation(ending_node, root_player_won)  
        stop_early, highest_uct, num_stable_iterations = agent_MCTS._check_early_stop(prev_highest_uct, epsilon, num_stable_iterations, stopping_iterations, confidence_interval)
        
        if stop_early:
            break
        else:
            prev_highest_uct = highest_uct

        
    child = max(agent_MCTS._root_node.children, 
                    key=lambda child: (child.winning_simulation_visits / child.simulation_visits 
                                       if child.simulation_visits > 0 and child.winning_simulation_visits > 0 
                                       else 0))
    move_played = agent_MCTS._position_played(player, child.board, current_board)
    
    #for child in agent_MCTS._root_node.children: 
        #print(f"Child Number: {agent_MCTS._root_node.children.index(child)}")
        #print(f"Visits: {child.simulation_visits}, Number winning visits: {child.winning_simulation_visits}, Win rate: {child.winning_simulation_visits / child.simulation_visits if child.simulation_visits > 0 and child.winning_simulation_visits > 0 else 0}\n")  
        
    agent_MCTS.write_output(move_played)
    
if __name__ == "__main__":
    main()
    
# Notes:

# - put all constants into the class
# - Now I'm wondering: in the simulation phase new children are generated at each level until a terminal node is reached. I use the '_generate_valid_boards' function, which only returns promising moves, to find the next possible moves during simulation. At the end of the day we want to find the best move for the root node's player. However, in the simulation phase promising moves are also returned to the opponent when its their turn to play. Is this the best way to do simulation to 


# notes: this MCTS algorithm version implemented with the following optimizations:
# - early stopping
# - UCT
# - promising moves
# - terminal state evaluation