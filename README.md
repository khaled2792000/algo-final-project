# Gomoku with Special Abilities

Gomoku with Special Abilities is an enhanced version of the classic Gomoku game, implemented using Python and Pygame. The game introduces new abilities such as moving pieces, removing rows or columns (referred to as "boom" abilities), and a minimax algorithm with alpha-beta pruning for AI moves.

## Features

- **Classic Gomoku Rules**: Win by placing five pieces in a row, column, or diagonal.
- **Special Abilities**:
  - Move a piece a limited number of times.
  - Remove a row or column of pieces after a certain number of rounds.
- **AI Player**: The game includes an AI opponent using the Minimax algorithm with alpha-beta pruning.
- **Dynamic Board Interaction**: Shake animation for invalid moves, hover highlights, and more.
- **Pygame Graphics**: Simple and interactive user interface using Pygame.

## Usage

Once the game is running, you can interact with the graphical interface to play. Use the mouse to select cells, move pieces, and activate special abilities. The game will guide you through your options based on the current game state.

## Gameplay

### Objective

The objective of Gomoku is to be the first player to form an unbroken line of five pieces horizontally, vertically, or diagonally.

### Controls

- **Placing Pieces**: Click on an empty cell to place your piece.
- **Moving Pieces**: Click the "Move Piece" button, then select a piece and click on an empty cell to move it.
- **Removing Rows/Columns**: Click the "Boom Row" or "Boom Column" button, then select the row or column you wish to remove.

### Game Rules

- Players take turns placing their pieces on the board.
- A line of five consecutive pieces wins the game.
- Each player has a limited number of moves and removals:
  - **Moves per Player**: 3 moves.
  - **Removes per Player**: 1 remove.
  - **Boom Limit Rounds**: Players can only use the remove action after 5 rounds.

## Code Structure

The project is organized into a single Python file, `GomokuGame.py`, which contains all the game logic, rendering, and AI implementation. Here's a breakdown of the key sections:

- **Initialization**: Setting up Pygame and defining constants.
- **Game State Variables**: Variables to track the state of the game.
- **Main Functions**: Functions to handle game logic, drawing, and user input.
- **AI Functions**: Implementation of the AI using the Minimax algorithm.
- **Special Abilities**: Functions to handle moving pieces and removing rows/columns.
- **Main Loop**: The game loop that continuously updates the game state and handles user input.

### Main Functions

- `reset_game()`: Resets the game state to its initial values.
- `print_board()`: Prints the current state of the board to the console for debugging.
- `draw_board()`: Draws the game board, pieces, and active animations.
- `draw_buttons()`: Draws the UI buttons and game information.
- `is_valid_move(row, col, player)`: Checks if a move is valid for a given player.
- `check_win(player)`: Checks if the given player has won the game.
- `handle_move_piece(row, col)`: Handles the logic for moving a piece on the board.
- `handle_boom(row, col, axis)`: Handles the logic for removing a row or column.

### AI Functions

- `minimax(...)`: The Minimax algorithm with Alpha-Beta pruning to determine the best move for the AI.
- `static_evaluation(matrix)`: Evaluates the current state of the board for the AI.
- `move_score(matrix, move, is_ai_turn)`: Calculates the score of a move for the given player.
- `bot_move()`: Executes the AI bot's move using the Minimax algorithm.

## AI Implementation

The AI opponent in this game uses the Minimax algorithm with Alpha-Beta pruning. This approach allows the AI to evaluate potential moves and choose the best one by simulating future game states. Here's a brief explanation of the key components:

- **Minimax Algorithm**: A recursive algorithm used to choose an optimal move for the AI by considering all possible moves and their outcomes.
- **Alpha-Beta Pruning**: An optimization technique for the Minimax algorithm that reduces the number of nodes evaluated by the AI, improving efficiency.
- **Static Evaluation**: A function that scores the board based on the positions of the pieces, helping the AI to decide on the best move.

## Special Abilities

### Move Piece

Players can move their pieces to an adjacent empty cell. This ability adds a layer of strategy, as players can reposition their pieces to block opponents or create new lines.

### Remove Rows/Columns

Players can remove entire rows or columns from the board. This powerful ability can drastically change the game state and is limited to one use per player after 5 rounds.
