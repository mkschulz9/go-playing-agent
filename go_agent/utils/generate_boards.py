from .helper_functions import undo_move, get_neighbors, count_liberties

# generates all valid boards that can be reached from the given board by the given player
# input: player, current board layout, previous board layout
# output: valid boards that can be reached
def generate_valid_boards(player, current_board, previous_board):
    valid_boards = []
    captured_stones = []

    positions = [(row, col) for row in range(5) for col in range(5)]
    for row, col in positions:
        if current_board[row][col] == 0:
            place_stone_capture(player, current_board, row, col, captured_stones)
            
            if not count_liberties(player, current_board, row, col) == 0 and not current_board == previous_board:
                valid_boards.append([row.copy() for row in current_board])
            
            undo_move(player, current_board, row, col, captured_stones)
            captured_stones.clear()

    return valid_boards


# places a stone for the given player and captures any opponent stones if they are left without liberties
# input: player, board to place stone on, row & column to place stone, list to keep track of captured stones
# output: none (modifies the board variable in-place)
def place_stone_capture(player, board, row, col, captured_stones):
    board[row][col] = player
    opponent = 3 - player

    for neighbor_row, neighbor_col in get_neighbors(row, col):
        if board[neighbor_row][neighbor_col] == opponent and count_liberties(opponent, board, neighbor_row, neighbor_col) == 0:
            capture_group(opponent, board, neighbor_row, neighbor_col, captured_stones)
    
    return len(captured_stones)

# capture an entire group of opponent's stones starting from the given cell
# input: player, board layout, row & column of cell, list to keep track of captured stones
# output: none (modifies the board variable in-place)
def capture_group(player_to_capture, board, start_row, start_col, captured_stones):
    to_capture = [(start_row, start_col)]
    visited = set()

    while to_capture:
        current_row, current_col = to_capture.pop()
        board[current_row][current_col] = 0
        captured_stones.append((current_row, current_col))
        visited.add((current_row, current_col))

        for neighbor_row, neighbor_col in get_neighbors(current_row, current_col):
            if board[neighbor_row][neighbor_col] == player_to_capture and (neighbor_row, neighbor_col) not in visited:
                to_capture.append((neighbor_row, neighbor_col))