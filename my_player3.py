
class MonteCarloTreeSearch:
    class BoardNode:
        def __init__(self, board, parent=None):
            self.board = board
            self.identifier = str(board)
            self.parent = parent
            self.children = []
            self.simulation_visits = 0
            self.winning_simulation_visits = 0

        def add_child(self, child_node):
            self.children.append(child_node)

        def update_values(self):
            pass
            
    def __init__(self, input_file_path):
        self.input_file_path = input_file_path
        self.expanded_nodes = {}
    
    # selects path, starting from root node and ending at leaf node
    def selection(self, board, player):
        possible_boards = self.generate_valid_boards(board, player)
        # check UCT values of each board and move to board with highest UCT value and continue algorithm process
    
    # generates all valid boards that can be reached from the given board by the given player
    # input: self, board layout, player number
    # output: list of valid moves
    def generate_valid_boards(self, board, player):
        valid_boards = []

        for row in range(5):
            for col in range(5):
                if board[row][col] == 0:
                    temporary_board = [row.copy() for row in board]
                    self.place_stone(temporary_board, row, col, player)
                    
                    if not self.is_suicide_move(temporary_board, row, col, player) and not self.is_ko_violation(temporary_board, board):
                        valid_boards.append(temporary_board)
        
        return valid_boards

    # places a stone for the given player and captures any opponent stones if they are left without liberties
    # input: self, board layout, row & column to place stone, player number
    def place_stone(self, board, row, col, player):
        board[row][col] = player
        self.capture_opponent_stones_if_any(board, row, col, 3 - player)

    # checks if placing a stone on the given cell results in a suicide move (a move that deprives the placed stone or its group of all liberties)
    # input: self, board layout, row & column to place stone, player number
    # output: true if a move is a suicide move, false if not 
    def is_suicide_move(self, board, row, col, player):
        return self.count_liberties(board, row, col, player) == 0

    # checks if making a move results in a board state that's identical to the previous state, which would be a KO violation
    # input: self, current board layout, previous board layout
    # output: true if a move is a KO volation, false if not
    def is_ko_violation(self, current_board, previous_board):
        for row in range(5):
            for col in range(5):
                if current_board[row][col] != previous_board[row][col]:
                    return False
        return True

    # counts the number of liberties (empty adjacent cells) for the stone/group starting from the given cell
    # input: self, board layout, row & column to start checking for liberties, player number
    # output: number of liberties
    def count_liberties(self, board, row, col, player):
        visited = set()
        to_check = [(row, col)]
        liberties = 0
        
        while to_check:
            current_row, current_col = to_check.pop()
            visited.add((current_row, current_col))
            
            for neighbor_row, neighbor_col in self.get_neighbors(current_row, current_col):
                if board[neighbor_row][neighbor_col] == 0:
                    liberties += 1
                elif board[neighbor_row][neighbor_col] == player and (neighbor_row, neighbor_col) not in visited:
                    to_check.append((neighbor_row, neighbor_col))
        
        return liberties

    # gets all valid neighboring cells for the given cell
    # input: self, row & column of cell
    # output: list of valid neighboring cells
    def get_neighbors(self, row, col):
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

    # captures any opponent's stones that are left without liberties due to the last move
    # input: self, board layout, row & column of last move, opponent number
    # output: none
    def capture_opponent_stones_if_any(self, board, row, col, opponent):
        for neighbor_row, neighbor_col in self.get_neighbors(row, col):
            if board[neighbor_row][neighbor_col] == opponent and self.count_liberties(board, neighbor_row, neighbor_col, opponent) == 0:
                self.capture_group(board, neighbor_row, neighbor_col, opponent)

    # capture an entire group of opponent's stones starting from the given cell
    # input: self, board layout, row & column of cell, opponent number
    # output: none
    def capture_group(self, board, start_row, start_col, opponent):
        to_capture = [(start_row, start_col)]
        visited = set()

        while to_capture:
            current_row, current_col = to_capture.pop()
            board[current_row][current_col] = 0
            visited.add((current_row, current_col))

            for neighbor_row, neighbor_col in self.get_neighbors(current_row, current_col):
                if board[neighbor_row][neighbor_col] == opponent and (neighbor_row, neighbor_col) not in visited:
                    to_capture.append((neighbor_row, neighbor_col))
    
    # expands leaf node
    def expansion(self):
        pass
    
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
        with open(self.input_file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]

        agent_color = int(lines[0])
        board_after_agent_move = [list(map(int, list(line))) for line in lines[1:6]]
        board_after_opponent_move = [list(map(int, list(line))) for line in lines[6:11]]
        file.close()

        return agent_color, board_after_agent_move, board_after_opponent_move

if __name__ == "__main__":
    agent_MCTS = MonteCarloTreeSearch("./input.txt")
    player, board_after_agent_move, board_after_opponent_move = agent_MCTS.readInput()
    print(f"Player: {player}\nBoard after agent's move:\n{board_after_agent_move}\nBoard after opponent's move:\n{board_after_opponent_move}\n")
    
    valid_boards = agent_MCTS.generate_valid_boards(board_after_opponent_move, player)
    print(f"Number of valid boards: {len(valid_boards)}\nValid Boards:")
    for board in valid_boards:
        print(f"Board {valid_boards.index(board) + 1}:")
        for row in board:
            print(row)
        print("\n")
    
    
    # read input file
    # pass current board layout to selection funciton
    
    # ToDo: finish making sure the generate valid boards function works correctly, and continue with slecetion function