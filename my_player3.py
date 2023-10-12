import math, random

class MonteCarloTreeSearch:
    class _Node:
        def __init__(self, player, board, parent, is_root=False):
            self.player = player
            self.board = board
            self.parent = parent
            self.children = []
            self.num_children_simulated = 0
            self.is_root = is_root
            self.simulated = False
            self.simulation_visits = 0
            self.winning_simulation_visits = 0
            
    def __init__(self, input_file_path):
        self._input_file_path = input_file_path
        self._root_node = None
        
    # sets the root node for the MCTS tree
    def set_root_node(self, player, current_board, previous_board):
        self._root_node = self._Node(player, current_board, previous_board, is_root=True)
    
    # selects a leaf node to expand (a node without children in the MCTS tree)
    # input: self
    # output: leaf node
    def selection(self):
        current_node = self._root_node
    
        while current_node.children:
            uct_values = [self._UCT(child) for child in current_node.children]
            current_node = current_node.children[uct_values.index(max(uct_values))]
            
        if current_node.is_root:
            possible_next_boards = self._generate_valid_boards(current_node.player, current_node.board, current_node.parent)
        else:
            possible_next_boards = self._generate_valid_boards(current_node.player, current_node.board, current_node.parent.board)
        next_player = 3 - current_node.player

        for board in possible_next_boards:
            child_node = self._Node(next_player, board, current_node)
            current_node.children.append(child_node)

        return current_node
    
    # expands leaf node
    # input: self, leaf node chosen during selection
    # output: next child node to start simulation from
    def expansion(self, leaf_node):
        if leaf_node.num_children_simulated == len(leaf_node.children) or self._is_terminal_state(leaf_node):
            return None
        
        unsimulated_children = [child for child in leaf_node.children if not child.simulated]
        chosen_child = random.choice(unsimulated_children)
        chosen_child.simulated = True
        leaf_node.num_children_simulated += 1
        
        return chosen_child
    
    # simulates game from leaf node
    def simulation(self):
        pass
    
    # backpropagates result of simulation, updating each node on path
    def backpropagation(self):
        pass

    # reads & parses input file
    # input: self
    # output: color number agent is playing as, board layout after agent's last move, board layout after opponent's last move
    def readInput(self):
        with open(self._input_file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]

        player = int(lines[0])
        previous_board = [list(map(int, list(line))) for line in lines[1:6]]
        current_board = [list(map(int, list(line))) for line in lines[6:11]]
        file.close()

        return player, current_board, previous_board
        
    # generates all valid boards that can be reached from the given board by the given player
    # input: self, player, current board layout, previous board layout
    # output: valid boards that can be reached
    def _generate_valid_boards(self, player, current_board, previous_board):
        valid_boards = []

        for row in range(5):
            for col in range(5):
                if current_board[row][col] == 0:
                    potential_board = [row.copy() for row in current_board]
                    self._place_stone_capture(player, potential_board, row, col)
                    
                    # check if move is not a suicide and not a ko violation
                    if not self._count_liberties(player, potential_board, row, col) == 0 and not self._is_ko_violation(potential_board, previous_board):
                        valid_boards.append(potential_board)
        
        return valid_boards

    # places a stone for the given player and captures any opponent stones if they are left without liberties
    # input: self, player to place stone for, board to place stone on, row & column to place stone
    # output: none (modifies the board variable in-place)
    def _place_stone_capture(self, player, board, row, col):
        board[row][col] = player
        opponent = 3 - player
        
        for neighbor_row, neighbor_col in self._get_neighbors(row, col):
            if board[neighbor_row][neighbor_col] == opponent and self._count_liberties(opponent, board, neighbor_row, neighbor_col) == 0:
                self._capture_group(opponent, board, neighbor_row, neighbor_col)

    # checks if making a move results in a board state that's identical to the previous state, which would be a KO violation
    # input: self, potential board layout, previous board layout
    # output: true if a move is a KO volation, false if not
    def _is_ko_violation(self, potential_board, previous_board):
        for row in range(5):
            for col in range(5):
                if potential_board[row][col] != previous_board[row][col]:
                    return False
        return True

    # counts the number of liberties (empty adjacent cells) for the stone/group starting from the given cell
    # input: self, player number, board layout, row & column of cell
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
    # input: self, player number, board layout, row & column of cell
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
    # input: self, row & column of cell
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
    def _UCT(self, node):
        C = 1.4
        w_i = node.winning_simulation_visits
        n_i = node.simulation_visits
        N_i = node.parent.simulation_visits if node.parent else 1
        
        if n_i == 0:
            return float('inf')
        
        uct_value = (w_i / n_i) + C * (math.sqrt(math.log(N_i) / n_i))
        return uct_value
    
    # checks if a board state is a terminal state
    def _is_terminal_state(self, node):
        # Check for completely filled board
        if all(all(cell != 0 for cell in row) for row in node.board):
            return True

        # Note: Checking for consecutive passes might need tracking of game history.
        
        return False

if __name__ == "__main__":
    #instantiate MCTS agent
    agent_MCTS = MonteCarloTreeSearch("./input.txt")
    
    # read and print input
    player, current_board, previous_board = agent_MCTS.readInput()
    print(f"Player: {player}\nBoard after agent's move:")
    for row in previous_board:
        print(row)
    print("Board after opponent's move:")
    for row in current_board:
        print(row)
    
    # set root node for MCTS tree
    agent_MCTS.set_root_node(player, current_board, previous_board)
    
    # for i in range(20):
    leaf_node = agent_MCTS.selection()
    child_node = agent_MCTS.expansion(leaf_node)

    print("\nLeaf node's board:")
    for row in leaf_node.board:
        print(row)
    
    print("\nChild node's board:")
    for row in child_node.board:
        print(row)