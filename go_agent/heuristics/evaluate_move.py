from ..utils.helper_functions import count_liberties, get_neighbors

# scores the given board based on the number of stones captured by the player
# input: current board layout, next board layout, player
# output: score of board for player
def capture_score(current_board, next_board, player):
    captured_stones = 0
    
    for x in range(5):
        for y in range(5):
            if current_board[x][y] == 3 - player and next_board[x][y] == 0:
                captured_stones += 1
                
    return captured_stones

# scores the boards based on the number of liberties gained by the player
# input: current board layout, next board layout, player
# output: difference in liberties gained by player
def liberty_score(current_board, next_board, player):
    current_liberties = calculate_liberties(current_board, player)
    next_liberties = calculate_liberties(next_board, player)
    
    return next_liberties - current_liberties

# scores the boards based on the number of liberties lost by the opponent
# input: current board layout, next board layout, player
# output: difference in liberties lost by opponent
def blocking_score(current_board, next_board, player):
    opponent = 3 - player
    current_opponent_liberties = calculate_liberties(current_board, opponent)
    next_opponent_liberties = calculate_liberties(next_board, opponent)
    
    return current_opponent_liberties - next_opponent_liberties

# determines how a move helps secure or extend the player's influence over certain areas of the board
# input: current board layout, next board layout, player
# output: difference in territory gained by player
def territorial_gain(current_board, next_board, player):
    opponent = 3 - player
    
    initial_territory = count_potential_territory(current_board, player)
    new_territory = count_potential_territory(next_board, player)
    
    initial_opponent_territory = count_potential_territory(current_board, opponent)
    new_opponent_territory = count_potential_territory(next_board, opponent)
    
    territorial_difference = (new_territory - initial_territory) - (new_opponent_territory - initial_opponent_territory)
    
    return territorial_difference

# determines if a move is in a corner or along an edge
# row and column of move placed
# value of move based on edge or corner placement
def corner_edge_score(row, col):
    score = 0
    
    if row in [0, 4] and col in [0, 4]:
        score += 2
    elif row in [0, 4] or col in [0, 4]:
        score += 1
        
    return score

# evaluates the overall 'influence' a stone could have on surrounding empty points
# board, row, and column placed
# value of stones 'influence'
def influence_score(board, row, col):
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
def atari_score(board, row, col, player):
    score = 0
    opponent = 3 - player
    
    for neighbor_row, neighbor_col in get_neighbors(row, col):
        if board[neighbor_row][neighbor_col] == opponent and count_liberties(opponent, board, neighbor_row, neighbor_col) == 1:
            score += 3
            
    return score

# evaluates the potential for future value of current placement
# board, row, column, player
# score of how well move sets up future play
def synergy_score(board, row, col, player):
    score = 0
    
    for neighbor_row, neighbor_col in get_neighbors(row, col):
        if board[neighbor_row][neighbor_col] == player:
            score += 2
            
    return score

# calculates the total number of liberties for the given player
# input: board layout, player
# output: total number of liberties for player
def calculate_liberties(board, player):
    total_liberties = 0
    
    for row in range(5):
        for col in range(5):
            if board[row][col] == player:
                total_liberties += count_liberties(player, board, row, col)
                
    return total_liberties

# counts the number of potential territories for the given player
# input: board layout, player
# output: number of potential territories for player
def count_potential_territory(board, player):
    territory_count = 0
    
    for row in range(5):
        for col in range(5):
            if board[row][col] == 0:
                neighbors = get_neighbors(row, col)
                if all(board[n_row][n_col] == player for n_row, n_col in neighbors):
                    territory_count += 1
                else:
                    break
                
    return territory_count