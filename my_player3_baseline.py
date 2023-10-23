import time, math, random

class MonteCarloTreeSearch:
    class _Node:
        def __init__(self, player, board, previous_board=None, parent_node=None, simulated = False):
            self.player = player
            self.board = board
            self.previous_board = previous_board
            self.parent_node = parent_node
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
        leaf_node.children = []
        possible_next_boards = self._generate_valid_boards(leaf_node.player, leaf_node.board, leaf_node.previous_board)
        
        for board in possible_next_boards:
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
            if current_node.player != player_won: # changed this from '==' to '!='
                current_node.winning_simulation_visits += 1
            current_node = current_node.parent_node
            
    # sets up root node and its children of the MCTS tree
    # input: player number, current board layout, previous board layout
    # Output none
    def init_tree(self, player, current_board, previous_board):
        self._root_node = self._Node(player, current_board, previous_board)
        possible_next_boards = self._generate_valid_boards(player, current_board, previous_board)
        next_player = self._opponent_player(player)
        
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
        
        if node.cached_uct is not None:
            return node.cached_uct
        
        C = math.sqrt(2)
        n_i = node.simulation_visits
        w_i = node.winning_simulation_visits
                
        if n_i == 0:
            return float('inf')
        
        uct_value = (w_i / n_i) + C * math.sqrt(log_parent_visits / n_i)
        node.cached_uct = uct_value
        
        return uct_value
            
    # generates all valid boards that can be reached from the given board by the given player
    # input: player, current board layout, previous board layout
    # output: valid boards that can be reached
    def _generate_valid_boards(self, player, current_board, previous_board):
        valid_boards = []
        
        for row, col in [(row, col) for row in range(5) for col in range(5)]:
            if current_board[row][col] == 0:
                original_state = current_board[row][col]
                self._place_stone_capture(player, current_board, row, col)
                
                if not self._count_liberties(player, current_board, row, col) == 0 and not current_board == previous_board:
                    valid_boards.append(self._copy_board(current_board))
                
                current_board[row][col] = original_state
                
        return valid_boards
    
    # places a stone for the given player and captures any opponent stones if they are left without liberties
    # input: player, board to place stone on, row & column to place stone, list to keep track of captured stones
    # output: none (modifies the board variable in-place)
    def _place_stone_capture(self, player, board, row, col):
        board[row][col] = player
        opponent = 3 - player

        for neighbor_row, neighbor_col in self._get_neighbors(row, col):
            if board[neighbor_row][neighbor_col] == opponent and self._count_liberties(opponent, board, neighbor_row, neighbor_col) == 0:
                self._capture_group(opponent, board, neighbor_row, neighbor_col)
            
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
    def _capture_group(self, player_to_capture, board, start_row, start_col):
        to_capture = [(start_row, start_col)]
        visited = set()

        while to_capture:
            current_row, current_col = to_capture.pop()
            board[current_row][current_col] = 0
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