def get_parameters():
    weights_UCT = {
        1: {'alpha': 0.001, 'beta': 0.001, 'gamma': 1, 'delta': 0.001},
        2: {'alpha': 0.01, 'beta': 0.01, 'gamma': 0.4, 'delta': 0.1},
        3: {'alpha': 0.1, 'beta': 0.1, 'gamma': 0.6, 'delta': 0.2}
    }    
    weights_move = {
        1: {'capture': 0.5, 'liberty': 1, 'blocking': 1, 'territory': 1.5, 'corner_edge': 1.2, 'influence': 1.5, 'atari': 0.5, 'synergy': 1},
        2: {'capture': 1.5, 'liberty': 1.5, 'blocking': 2, 'territory': 1, 'corner_edge': 0.5, 'influence': 1, 'atari': 1, 'synergy': 1.5},
        3: {'capture': 3, 'liberty': 1.5, 'blocking': 1, 'territory': 2, 'corner_edge': 0.5, 'influence': 0.5, 'atari': 2, 'synergy': 1}
    }
    min_max_values_move = {
        'capture': [0, 21],
        'liberty': [-134, 224],
        'blocking': [0, 78],
        'territory': [-3, 5],
        'corner_edge': [0, 2],
        'influence': [0, 8],
        'atari': [0, 9],
        'synergy': [0, 8]
    }
    weights_board = {
        1: {'player_score': 2, 'defensive': 0.5, 'suicides': 0.5, 'captured_spaces': 2, 'eyes': 1, 'alive_groups': 0.5},
        2: {'player_score': 1.5, 'defensive': 2, 'suicides': 0.5, 'captured_spaces': 1.5, 'eyes': 1, 'alive_groups': 1},
        3: {'player_score': 3, 'defensive': 1.5, 'suicides': 0.5, 'captured_spaces': 1, 'eyes': 1, 'alive_groups': 2}
    }
    min_max_values_board = {
        'player_score': [-24.5, 19.5],
        'defensive': [0, 6],
        'suicides': [0, 24],
        'captured_spaces': [0, 8],
        'eyes': [0, 8],
        'alive_groups': [0, 2],
    }
    weights_final_move = {
    1: (0.001, 0.01),
    2: (0.5, 0.1),    
    3: (0.7, 0.5)  
    }
    
    move_multipliers = {'capture': 3, 'territory': 2, 'atari': 2, 'influence': 2}
    board_multipliers = {'player_score': 3, 'captured_spaces': 2}

    settings = {
        'weights_UCT': weights_UCT,
        'weights_move': weights_move,
        'min_max_values_move': min_max_values_move,
        'weights_board': weights_board,
        'min_max_values_board': min_max_values_board,
        'weights_final_move': weights_final_move,
        'move_multipliers': move_multipliers,
        'board_multipliers': board_multipliers
    }

    return settings