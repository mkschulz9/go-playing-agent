class _Node:
    def __init__(self, player, board, previous_board=None, parent_node=None, simulated = False):
        self.player = player
        self.board = board
        self.previous_board = previous_board
        self.parent_node = parent_node
        self.children = []
        self.move_score = 0
        self.board_score = 0
        self.total_value = 0
        self.simulated = simulated
        self.simulation_visits = 0
        self.winning_simulation_visits = 0