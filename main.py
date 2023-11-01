import time
from go_agent.mcts import MonteCarloTreeSearch
from go_agent.utils.helper_functions import (get_game_stage, count_pieces, 
                              get_opening_move, write_output, 
                              read_input, position_played)

def main():
    start_time = time.time()
    max_time = 7
    
    # initialize agent
    agent_MCTS = MonteCarloTreeSearch()
    player, current_board, previous_board = read_input("./io/input.txt")
    agent_MCTS.init_tree(player, current_board, previous_board)
    game_stage = get_game_stage(current_board)
    
    # check if the game is in the opening stage, return optimal move if so
    if count_pieces(current_board) < 2:
        move_played = get_opening_move(player, current_board)
        write_output(move_played)
        exit()
        
    # adjust weights for player 1 to play more aggressively
    if player == 1:
        agent_MCTS.update_player_1_weights()

    # main MCTS loop
    while time.time() - start_time < max_time:
        leaf_node = agent_MCTS.selection()        
        child_node = agent_MCTS.expansion(leaf_node)        
        ending_node = agent_MCTS.simulation(child_node)
        agent_MCTS.backpropagation(ending_node)

    # find the child of root node with the maximum value ending value and play that move
    child = max(agent_MCTS._root_node.children, key=lambda child: agent_MCTS.calc_child_value(child, game_stage))
    move_played = position_played(player, child.board, current_board)
    write_output(move_played)
    
if __name__ == "__main__":
    main()