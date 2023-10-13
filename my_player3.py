import math, random

class MonteCarloTreeSearch:
    class _Node:
        def __init__(self, player, board, previous_board=None, parent=None):
            self.player = player
            self.board = board
            self.previous_board = previous_board
            self.parent = parent
            self.children = []
            self.simulated = False
            self.simulation_visits = 0
            self.winning_simulation_visits = 0
            
    def __init__(self, input_file_path):
        self._input_file_path = input_file_path
        self._root_node = None
        
    # sets the root node for the MCTS tree
    # input: self, player number, current board layout, previous board layout
    # Output none (modifies the root node)
    def set_root_node(self, player, current_board, previous_board):
        self._root_node = self._Node(player, current_board, previous_board)
        
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
    
    # selects a leaf node to expand (a node without children in the MCTS tree)
    # input: self
    # output: leaf node
    def selection(self):
        current_node = self._root_node
        while current_node.children:
            uct_values = [self._UCT(child) for child in current_node.children]
            current_node = current_node.children[uct_values.index(max(uct_values))]
            
        return current_node
    
    # expands leaf node
    # input: self, leaf node chosen during selection
    # output: next child node to start simulation from
    def expansion(self, leaf_node):
        if not leaf_node.children:
            possible_next_boards = self._generate_valid_boards(leaf_node.player, leaf_node.board, leaf_node.previous_board)
            next_player = 3 - leaf_node.player
            for board in possible_next_boards:
                child_node = self._Node(next_player, board, leaf_node.board, leaf_node)
                leaf_node.children.append(child_node)
        if leaf_node.children:
            unsimulated_children = [child for child in leaf_node.children if not child.simulated]
            if unsimulated_children:
                chosen_child = random.choice(unsimulated_children)
                chosen_child.simulated = True
                return chosen_child
        return None
    
    # simulates game randomly from the given node
    # input: self, node to start simulation from
    # output: node at the end of the simulation
    def simulation(self, starting_node):
        current_node = starting_node
        
        while True:
            possible_next_boards = self._generate_valid_boards(current_node.player, current_node.board, current_node.previous_board)
            if not possible_next_boards:
                root_player_won = self._has_root_player_won(current_node.board)
                return current_node, root_player_won
            
            next_board = random.choice(possible_next_boards)
            
            # Determine the move played
            #move_played = self.position_played(current_node.player, next_board, current_node.board)
            #if move_played:
                #print(f"\nPlayer {current_node.player} played move at {move_played}. Board:")
                #for row in next_board:
                    #print(row)
           
            next_player = 3 - current_node.player
            
            new_child = self._Node(next_player, next_board, current_node.board, current_node)
            current_node.children.append(new_child)
            current_node = new_child
                
    # returns the position played by the player       
    def position_played(self, player, current_board, previous_board):
        for row in range(5):  # assuming the board is of size 5x5
            for col in range(5):
                if current_board[row][col] == player and previous_board[row][col] != player:
                    return (row, col)
        return None  # if no position was found (this shouldn't happen if the game is played correctly)

    # backpropagates result of simulation, updating each node on path
    def backpropagation(self):
        pass
    
    # determines if the root player has won the game
    # input self, board layout at end of game
    # output: True if root player has won, False otherwise
    def _has_root_player_won(self, ending_board):
        komi_value = 2.5

        # Calculate scores for each player
        player1_score = sum(row.count(1) for row in ending_board)
        player2_score = sum(row.count(2) for row in ending_board) + komi_value
        # print(f"\n\nPlayer 1 score: {player1_score} & Player 2 score: {player2_score}")
        
        if self._root_node.player == 1:
            root_score = player1_score
            opponent_score = player2_score
        else:
            root_score = player2_score
            opponent_score = player1_score
        
        return root_score > opponent_score
        
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
                    if not self._count_liberties(player, potential_board, row, col) == 0 and not potential_board == previous_board:
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

if __name__ == "__main__":
    #instantiate MCTS agent
    agent_MCTS = MonteCarloTreeSearch("./input.txt")
    
    # read and print input
    player, current_board, previous_board = agent_MCTS.readInput()
    print(f"Agent Player: {player}\nBoard after agent's move:")
    for row in previous_board:
        print(row)
    print("\nBoard after opponent's move:")
    for row in current_board:
        print(row)
    
    # set root node for MCTS tree
    agent_MCTS.set_root_node(player, current_board, previous_board)
    
    # for i in range(20):
    leaf_node = agent_MCTS.selection()
    child_node = agent_MCTS.expansion(leaf_node)
    ending_node, root_player_won = agent_MCTS.simulation(child_node)