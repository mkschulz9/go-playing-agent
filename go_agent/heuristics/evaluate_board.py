from ..utils.helper_functions import count_liberties, get_neighbors

# counts the number of groups of stones with high liberties for the given player
# input: board layout, player
# output: number of groups of stones with high liberties for the given player
def defensive_structures(board, player):
    high_liberty_count = 0
    visited = set()

    for row in range(5):
        for col in range(5):
            if (row, col) not in visited and board[row][col] == player:
                group_liberties = count_liberties(player, board, row, col)
                
                if group_liberties >= 2:
                    high_liberty_count += 1

                to_check = [(row, col)]
                while to_check:
                    current_row, current_col = to_check.pop()
                    visited.add((current_row, current_col))
                    
                    for neighbor_row, neighbor_col in get_neighbors(current_row, current_col):
                        if board[neighbor_row][neighbor_col] == player and (neighbor_row, neighbor_col) not in visited:
                            to_check.append((neighbor_row, neighbor_col))

    return high_liberty_count

# counts the number of potential suicides for the root player's opponent
# input: board layout, opponent player number
# output: number of potential suicides for opponent
def count_potential_suicides(board, opponent):
    potential_suicides = 0

    for row in range(5):
        for col in range(5):
            if board[row][col] == 0:
                neighbors = get_neighbors(row, col)
                
                if all(board[r][c] != opponent for r, c in neighbors):
                    potential_suicides += 1

    return potential_suicides

# counts the number of spaces captured by the given player
# input: board layout, player
# output: number of spaces captured by player
def count_captured_spaces(board, player):
    captured_spaces = 0
    visited = set()

    for row in range(5):
        for col in range(5):
            if board[row][col] == 0 and (row, col) not in visited:
                is_captured, space_count, space_visited = is_space_captured(board, row, col, player)
                if is_captured:
                    captured_spaces += space_count
                visited.update(space_visited)

    return captured_spaces

# counts the number of total eyes for the given player
# input: board layout, player
# output: number of total eyes for the given player
def count_total_eyes(board, player):
    eye_count = 0

    for row in range(5):
        for col in range(5):
            if board[row][col] == 0:
                neighbors = get_neighbors(row, col)
                if all(board[r][c] == player for r, c in neighbors):
                    eye_count += 1

    return eye_count

# counts number of groups with at least two eyes
# input: board layout, player
# output: number of groups with at least two eyes
def count_alive_groups(board, player):
    alive_groups = 0
    visited = set()
    
    for row in range(5):
        for col in range(5):
            if (row, col) not in visited and board[row][col] == player:
                group = []
                to_check = [(row, col)]
                
                while to_check:
                    current_row, current_col = to_check.pop()
                    visited.add((current_row, current_col))
                    group.append((current_row, current_col))
                    
                    for neighbor_row, neighbor_col in get_neighbors(current_row, current_col):
                        if board[neighbor_row][neighbor_col] == player and (neighbor_row, neighbor_col) not in visited:
                            to_check.append((neighbor_row, neighbor_col))
                
                if count_eyes(board, group) >= 2:
                    alive_groups += 1
                    
    return alive_groups

# determines if the given space is captured by the given player
# input: board layout, row & column of space, player
# output: boolean indicating if space is captured, number of spaces captured, set of visited spaces
def is_space_captured(board, start_row, start_col, player):
    to_check = [(start_row, start_col)]
    visited = set()
    is_captured = True
    space_count = 0

    while to_check:
        current_row, current_col = to_check.pop()
        visited.add((current_row, current_col))
        space_count += 1

        for neighbor_row, neighbor_col in get_neighbors(current_row, current_col):
            if board[neighbor_row][neighbor_col] == 0 and (neighbor_row, neighbor_col) not in visited:
                to_check.append((neighbor_row, neighbor_col))
            elif board[neighbor_row][neighbor_col] != player:
                is_captured = False

    return is_captured, space_count, visited

# counts the number of eyes for the given group
# input: board layout, group of stones
# output: number of eyes for the given group
def count_eyes(board, group):
    eye_count = 0
    
    for row, col in group:
        neighbors = get_neighbors(row, col)
        if all(board[r][c] == board[row][col] for r, c in neighbors):
            eye_count += 1
            
    return eye_count