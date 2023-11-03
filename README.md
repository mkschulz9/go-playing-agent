# CSCI-561: Foundations of Artificial Intelligence

## Assignment #2: 5x5 Go Playing Agent

Welcome to the repository for Assignment #2 for CSCI-561, USC's Foundations of Artificial Intelligence graduate course. This project implements a 5x5 Go playing agent using a modified Monte Carlo Tree Search Algorithm.

---

## Table of Contents
1. [Introduction](#introduction)
    - [Traditional Go](#traditional-go)
    - [Monte Carlo Tree Search (MCTS)](#monte-carlo-tree-search-mcts)
    - [5x5 Go](#5x5-go)
    - [Input/Output](#inputoutput)
2. [Implementation](#implementation)
    - [Main](#main-mainpy)
    - [Go Agent](#go-agent-go_agent)
        - [Heuristics](#heuristics-heuristics)
            - [Evaluate Move](#evaluate-move-evaluate_movepy)
            - [Evaluate Board](#evaluate-board-evaluate_boardpy)
        - [Utils](#utils-utils)
            - [Generate Boards](#generate-boards-generate_boardspy)
            - [Helper Functions](#helper-functions-helper_functionspy)
        - [MCTS](#mcts-mctspy)
        - [MCTS Config](#mcts-config-mcts_configpy)
        - [Node](#node-nodepy)
3. [Getting Started: Running the Program](#getting-started-running-the-program)

---

## Introduction

### Traditional Go
- Go is an ancient board game that originated in China over 4,500 years ago. The game's objective is to claim more territory on the board than your opponent by placing and capturing stones. To learn more, [click here](https://en.wikipedia.org/wiki/Go_(game)).

---

### Monte Carlo Tree Search (MCTS)
- Monte Carlo Tree Search (MCTS) is a search algorithm commonly used in decision-making problems, especially in games like Go and Chess. Here's a brief overview:

    1. Selection: Start from the root of the tree and traverse down to a leaf node by selecting the node that has the highest value according to a specific policy (usually the UCT formula).

    2. Expansion: Once a leaf node is reached, expand it by adding one or more child nodes representing possible moves or decisions.

    3. Simulation: Perform a random simulation (rollout) from one of the child nodes until a terminal state (like the end of the game) is reached.

    4. Backpropagation: Once the simulation is complete, propagate the result (win or loss) back up the tree, updating the value and visit count of each node along the path.

- The process repeats, selecting nodes to expand and simulating outcomes until a computational budget (like time or number of simulations) is reached. The move with the best value or the most visits at the root is then chosen as the best move. To learn more about MCTS, [click here](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search).

---

### 5x5 Go
- While Go is typically played on a 19x19 grid, the 5x5 version simplifies it to a smaller board with only 5 squares in each dimension. The basic rules of this version, which the agent follows, are:
    1. Players aren't allowed to play "suicide moves," meaning stones cannot be placed such that the placed stone or its unit has no liberties. 
    2. A KO cannot be recaptured.
    3. A maximum of 24 moves can be played per game. 
    4. Once the game is complete, the player with the highest score is determined the winner:
        - Black's Score = # of stones on the board
        - White's Score = # of stones on the board + komi value of 2.5
 > All other rules follow traditional Go rules.

---

 ### Input/Output
 #### Input
 - The program expects to read a text input in the following format:
    - **Line 1:** Value of "1" or "2" denoting the color the agent is playing as (1 = Black; 2 = White).
    - **Lines 2-6:** Board layout after agent's last move. 
    - **Lines 7-11:** Board layout after opponent's last move. 
- Note: In lines 2-11, "1"s represent Black's stones, "2"s represent White's stones, and "0"s represent empty spaces. 

#### Output
- Once the agent has decided on the move it wants to play, it will generate a file titled "output.txt" in the "io" directory denoting the position it wants to place its stone in the following format:
    - **Line 1:** [row #],[column #] (0 indexed)

---

## Implementation 
- This project's implementation of MCTS differs in several ways from the base algorithm. Here are some of the key differences:
    1. Heuristic Functions: 
        - Two heuristic functions are used: one to evaluate the value of a move for a player and another to evaluate the value of a board for a player. The move evaluation heuristic guides MCTS in searching more "promising" nodes. Helping to prune the search space and find the best move faster. The board evaluation heuristic is used to score the board at the end of simulations, where the score is then used during backpropagation. 
    2. Modified UCT Function:
        - The UCT function is modified to take into account the move and board scores for a node. This helps the agent select the best node to expand.
    3. Weights:
        - Weights have been added throughout the program, which can be fine-tuned to adjust the playing style of the agent. They are also designed to change based on the stage of the game, which can further improve the agent's performance.
    4. Choosing Final Move:
        - The final move is chosen using a combination of the ratio of winning simulations to total simulations and the value of the board to total simulations. This helps the agent choose the best move, even if it hasn't been simulated as much as other moves.

> Note: Feel free to visit each link below for a more in-depth look at the code with additional descriptions!

---

### Main ([`main.py`](https://github.com/mkschulz9/go-playing-agent/blob/main/main.py))
- This file orchestrates the execution of the program. Its main responsibilities include:
    1. Initializing the agent and reading the input file.
    2. Determining if the game is in an opening stage and returning an optimal move if so.
    3. Adjusting weights to play more aggressively as Black.
    4. Running the MCTS loop (Selection, Expansion, Simulation, and Backpropagation) until the computational budget is reached.
    5. Choosing the optimal move and writing the move to a file.

---

### Go Agent ([`go_agent`](https://github.com/mkschulz9/go-playing-agent/tree/main/go_agent))

---

#### Heuristics ([`heuristics`](https://github.com/mkschulz9/go-playing-agent/tree/main/go_agent/heuristics))

- #### Evaluate Move ([`evaluate_move.py`](https://github.com/mkschulz9/go-playing-agent/blob/main/go_agent/heuristics/evaluate_move.py))
    - This file contains the heuristic functions used to determine the value of a move for a player.

- #### Evaluate Board ([`evaluate_board.py`](https://github.com/mkschulz9/go-playing-agent/blob/main/go_agent/heuristics/evaluate_board.py))
    - This file contains the heuristic functions used to determine the value of a board for a player.

---

#### Utils ([`utils`](https://github.com/mkschulz9/go-playing-agent/tree/main/go_agent/utils))

- #### Generate Boards ([`generate_boards.py`](https://github.com/mkschulz9/go-playing-agent/blob/main/go_agent/utils/generate_boards.py))
    - This is where the logic for generating all possible boards given a board and the current player is held.

- #### Helper Functions ([`helper_functions.py`](https://github.com/mkschulz9/go-playing-agent/blob/main/go_agent/utils/helper_functions.py))
    - This file contains helper functions used throughout the program.

---

#### MCTS ([`mcts.py`](https://github.com/mkschulz9/go-playing-agent/blob/main/go_agent/mcts.py))
- This file holds the `MonteCarloTreeSearch` class, which defines a custom Monte Carlo Tree Search Algorithm for the 5x5 version of Go. The methods include:
    1. `init_tree`
    2. `selection`
    3. `expansion`
    4. `simulation`
    5. `backpropagation`
    6. `update_player_1_weights`
    7. `calc_child_value`
    8. `_modified_UCT`
    9. `_evaluate_move`
    10. `_evaluate_board`

---

#### MCTS Config ([`mcts_config.py`](https://github.com/mkschulz9/go-playing-agent/blob/main/go_agent/mcts_config.py))
- This file configures various hyperparameters for the agent, which control its playing style.

---

#### Node ([`node.py`](https://github.com/mkschulz9/go-playing-agent/blob/main/go_agent/node.py))
- This file contains the `_Node` class, which represents a node in the tree. It encapsulates a particular game state and provides attributes required for game tree evaluations. These attributes include:
    1. `player:` the player playing the move
    2. `board:` the board state
    3. `previous_board:` the previous board state
    4. `parent_node:` the parent node
    5. `children:` the node's children nodes
    6. `move_score:` the value of moving into this node from the parent node from the perspective of the parent node's player
    7. `board_score:` the value of the board from the perspective of the root node's player
    8. `total_value:` the total value of the node aggregated during backpropagation
    9. `simulated:` a boolean indicating if the node has been simulated
    10. `simulation_visits:` the number of times the node has been visited during the simulations
    11. `winning_simulation_visits:` the number of times the node has been visited during a winning simulation for the parent's player

---

## Getting Started: Running the Program

- Here's how to clone the repository and run the main program on your local machine.

### 1. Clone the Repository

```bash
git clone [repository-URL]
```

### 2. Navigate to the Directory

```bash
cd [repository-name]
```

### 3. Run the Main File

- Execute the main program with the following command:

```bash
python main.py
```

> Now, the program should be up and running!
