import math, random, time

class MonteCarloTreeSearch:
    class _Node:
        def __init__(self, player, board, previous_board=None, parent_node=None, simulated = False):
            self.player = player
            self.board = board
            self.previous_board = previous_board
            self.parent_node = parent_node
            self.children = []
            self.value = 0
            self.simulated = simulated
            self.simulation_visits = 0
            self.winning_simulation_visits = 0
            
    def __init__(self, input_file_path):
        self._input_file_path = input_file_path
        self._root_node = None
        #self.time_limit = time_limit
        #self.quality_threshold = quality_threshold
        #self.start_time = 0 
        
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
        leaf_node.children = []
        possible_next_boards = self._generate_valid_boards(leaf_node.player, leaf_node.board, leaf_node.previous_board)
        next_player = 3 - leaf_node.player
        
        for board in possible_next_boards:
            child_node = self._Node(next_player, board, leaf_node.board, leaf_node)
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
        
        while True:
            valid_boards = self._generate_valid_boards(current_node.player, current_node.board, current_node.previous_board)
            
            if valid_boards == [] or depth == estimated_moves_remaining:
                return current_node, self._has_root_player_won(current_node.board)
            
            random_board = random.choice(valid_boards) # instead of generating random board, generate random moves (higher cost to store boards)
            next_player = 3 - current_node.player
            
            new_child = self._Node(next_player, random_board, current_node.board, current_node)
            current_node.children.append(new_child)
            current_node = new_child
            depth += 1

    # backpropagates result of simulation, updating each node on path
    # input: node at end of simulation, boolean indicating if root player won
    # output: none (modifies nodes attributes in winning path)
    def backpropagation(self, ending_node, root_player_won):
        value = self._evaluate_terminal_board(ending_node)
        current_node = ending_node
        
        if root_player_won:
            player_won = self._root_node.player
        else:
            player_won = 3 - self._root_node.player

        while current_node.parent_node:
            current_node.simulation_visits += 1
            if current_node.player == player_won:
                current_node.winning_simulation_visits += 1
            current_node.value += value
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

    # estimates the number of moves remaining in the game
    # input: board layout
    # output: estimated number of moves remaining
    def _estimate_remaining_moves(self, board):
        remaining_moves = 24
        num_stones = sum(cell != 0 for row in board for cell in row)
        remaining_moves -= num_stones        
        captured_stones = 0
        
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == 0:
                    neighbors = self._get_neighbors(row, col)
                    
                    if all(board[n_row][n_col] == neighbors[0][1] for n_row, n_col in neighbors):
                        liberties = self._count_liberties(board[neighbors[0][0]][neighbors[0][1]], board, neighbors[0][0], neighbors[0][1])
                        if liberties == len(neighbors):
                            captured_stones += 1
        
        remaining_moves -= captured_stones

        if remaining_moves < 0:
            remaining_moves = 0
        
        return remaining_moves
    
    # scores the given terminal board for the root player based on the number of stones and eyes of root player
    # input: terminal board layout
    # output: score of board for root player
    def _evaluate_terminal_board(self, ending_node):
        # Notes: multiplying # stones by 5; multiplying eyes by 2; and multiplying opp suicides by 10 produced good result on Vocareum;
        # I think number of stones on the board is the best way to guage value of the board
        score = 0
        root_player = self._root_node.player
        opponent = 3 - root_player
        #komi_value = 2.5
        
        # count the number of root player's stones on the board (highest weight)
        root_player_stones = sum(row.count(root_player) for row in ending_node.board)        
        # factor in the root player's stone groups with high liberties to ensure they are not easily captured in future turns
        groups_with_high_liberties = self._defensive_structures(ending_node.board, root_player)
        # potential suicides for spponent: While a bit more situational, recognizing areas where the opponent might be forced into bad moves could also be advantageous.
        if self._check_equal_pieces(ending_node.board):
            score += self._count_potential_suicides(ending_node.board, opponent)
        # determine number of spaces captured by root player
        captured_spaces = self._count_captured_spaces(ending_node.board, root_player)
        #score += self._count_eyes(ending_node.board, root_player) * 2
        
        score += root_player_stones
        score += groups_with_high_liberties
        score += captured_spaces
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

    # finds the number of open spaces next to a group of stones of the given player
    # input: board layout, player
    # output: number of eyes for the given player
    def _count_eyes(self, board, player):
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
    
    # returns the position played by the player
    # input: player, current board layout, previous board layout
    # output: position played by the player    
    def _position_played(self, player, current_board, previous_board):
        for row in range(5):
            for col in range(5):
                if current_board[row][col] == player and previous_board[row][col] != player:
                    return (row, col)
        return None
        
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
    def _UCT(self, node, log_parent_visits):
        C = 2
        n_i = node.simulation_visits
        
        if n_i == 0:
            return float('inf')
        
        w_i = node.winning_simulation_visits
        uct_value = (w_i / n_i) + C * math.sqrt(log_parent_visits / n_i)
        return uct_value

def main():
    agent_MCTS = MonteCarloTreeSearch("./input.txt")
    player, current_board, previous_board = agent_MCTS.read_input()
    agent_MCTS.init_tree(player, current_board, previous_board)
    end_time = time.time() + 10
    
    while time.time() < end_time:
        leaf_node = agent_MCTS.selection()        
        child_node = agent_MCTS.expansion(leaf_node)        
        ending_node, root_player_won = agent_MCTS.simulation(child_node)
        agent_MCTS.backpropagation(ending_node, root_player_won)
            
    child = max(agent_MCTS._root_node.children, key=lambda child: child.value)
    move_played = agent_MCTS._position_played(player, child.board, current_board)
    agent_MCTS.write_output(move_played)
    
if __name__ == "__main__":
    main()
    
# Notes:
# - when playing as Black, I might need to be more aggressive at capturing stones
# - Also, I think the number of opponent pieces captured during a rollout can be beneficial, how can I implement this in my code?
# - eval function using first two metrics seemed to work best (excluding opp suicide moves)
# - don't forget to implement time constraint too
# - best known moves for beginning and ending game
# - handle if input has 'PASS'
# - number of eyes is important
# - verify all funcitons work correctly

# Left Off:
# going to implement a depth cutoff on simulaiton since game ends after 24 moves -> how to determine the number of moves completed on a board?
# finish testing if current moves remaining funciton works
# just finished redoing eval funciton, about to test (seems no depth cutoff worked best)