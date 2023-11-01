import random
from collections import deque

# reads & parses input file
# input: None
# output: color number agent is playing as, board layout after agent's last move, board layout after opponent's last move
def read_input(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    player = int(lines[0])
    previous_board = [list(map(int, list(line))) for line in lines[1:6]]
    current_board = [list(map(int, list(line))) for line in lines[6:11]]

    return player, current_board, previous_board

# writes output to file on first file in format: 'row,column'
# input: position played by agent
# output: none (writes to file)
def write_output(move_played):
    with open('./io/output.txt', 'w') as file:
        file.write(f"{move_played[0]},{move_played[1]}")
        
# places stone for player in optimal position when beginning the game
# input: player number, current board layout
# output: position to place stone
def get_opening_move(player, current_board):
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
                
# counts number of pieces on the board
# input: board layout
# output: number of pieces on the board
def count_pieces(board):
    number_pieces = 0
    
    for row in board:
        for cell in row:
            if cell != 0:
                number_pieces += 1
                
    return number_pieces

# undoes a move by the given player and restores the board to its original state
# input: player, board to undo move on, row & column to undo move, list of captured stones to restore
# output: none (modifies the board variable in-place)
def undo_move(player, board, row, col, captured_stones):
    board[row][col] = 0
    
    for r, c in captured_stones:
        board[r][c] = 3 - player
        
# gets all valid neighboring cells for the given cell
# input: row & column of cell
# output: list of valid neighboring cells
def get_neighbors(row, col):
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

# determines if the number of stones for each player is within 2 of each other
# input: board layout
# output: boolean indicating if the number of stones for each player is within 2 of each other
def check_equal_pieces(board):
    count_player1 = 0
    count_player2 = 0

    for row in board:
        for cell in row:
            if cell == 1:
                count_player1 += 1
            elif cell == 2:
                count_player2 += 1

    return abs(count_player1 - count_player2) <= 2

# calculates the "depth" from a given point to the nearest stone of the opponent player on the board
# input: point to calculate depth from, board layout, player
# output: depth from point to nearest stone of opponent player
def calculate_depth_from_player_stones(point, board, player):
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

# calculates scores of each opponent
# board layout
# scores for each player
def get_player_scores(board):
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

# estimates the number of moves remaining in the game
# input: board layout
# output: estimated number of moves remaining
def estimate_remaining_moves(board):
    total_moves = 24
    player1_stones, player2_stones  = get_player_scores(board)
    player2_stones -= 2.5
    
    if player1_stones == player2_stones:
        remaining_moves = total_moves - (player1_stones + player2_stones)
    elif player1_stones > player2_stones:
        remaining_moves = total_moves - (player1_stones * 2)
    else:
        remaining_moves = total_moves - (player2_stones * 2)
    
    return remaining_moves

# returns the stage of the game based on the number of moves remaining
# input: board layout
# output: stage of game (1, 2, or 3)
def get_game_stage(board):
    moves_remaining = estimate_remaining_moves(board)
    
    if moves_remaining > 16:
        return 1
    elif moves_remaining > 8:
        return 2
    else:
        return 3
    
# counts the number of liberties (empty adjacent cells) for the stone/group starting from the given cell
# input: player, board layout, row & column of cell
# output: number of liberties of the stone/group
def count_liberties(player, board, row, col):
    visited = set()
    to_check = [(row, col)]
    liberties = 0
    
    while to_check:
        current_row, current_col = to_check.pop()
        visited.add((current_row, current_col))
        
        for neighbor_row, neighbor_col in get_neighbors(current_row, current_col):
            if board[neighbor_row][neighbor_col] == 0:
                liberties += 1
            elif board[neighbor_row][neighbor_col] == player and (neighbor_row, neighbor_col) not in visited:
                to_check.append((neighbor_row, neighbor_col))
    
    return liberties

# returns the position played by the player
# input: player, current board layout, previous board layout
# output: position played by the player    
def position_played(player, current_board, previous_board):
    for row in range(5):
        for col in range(5):
            if current_board[row][col] == player and previous_board[row][col] != player:
                return (row, col)
            
    return None

# determines iplayer that won the game
# input: board layout at end of game
# output: player that won the game
def player_won(board):
    player1_score, player2_score = get_player_scores(board)

    return 1 if player1_score > player2_score else 2

# calculates the final winning simulation ratio and value ratio for a child node
# input: alpha, beta, child node
# output: winning simulation ratio, value ratio
def calculate_ratios(alpha, beta, child):
    win_sim_ratio = alpha * ((child.winning_simulation_visits / child.simulation_visits) * 100) if child.winning_simulation_visits and child.simulation_visits else 0
    value_sim_ratio = beta * (child.total_value / child.simulation_visits) if child.total_value and child.simulation_visits else 0

    return win_sim_ratio, value_sim_ratio