import pygame
import sys
import threading
import time
import math
import copy  # For creating deep copies of objects
from functools import lru_cache  # Least-recently-used cache decorator

# Initialize Pygame
pygame.init()

# Screen dimensions and grid size
SCREEN_SIZE = 600  # Size of the game screen
GRID_SIZE = 6  # Size of the grid (6x6)
CELL_SIZE = SCREEN_SIZE // GRID_SIZE  # Size of each cell in the grid
BUTTON_WIDTH = 200  
BUTTON_HEIGHT = 40 
MODAL_WIDTH = 400  
MODAL_HEIGHT = 200  

# Colors
WHITE = (255, 255, 255)  
BLACK = (0, 0, 0) 
RED = (255, 0, 0) 
YELLOW = (255, 255, 0) 
GRAY = (200, 200, 200)  
BLUE = (0, 0, 255) 
GREEN = (0, 255, 0)  

# Game settings
MOVES_PER_PLAYER = 3 
REMOVES_PER_PLAYER = 1 
BOOM_LIMIT_ROUNDS = 5  

# Game state variables
board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # Game board initialized to 0
player_turn = 1  # Player 1 starts the game
moves_left = {1: MOVES_PER_PLAYER, 2: MOVES_PER_PLAYER}  # Moves left for each player
removes_left = {1: REMOVES_PER_PLAYER, 2: REMOVES_PER_PLAYER}  # Removes left for each player
boom_used = {1: False, 2: False}  # Track if boom action has been used by each player
last_move = {1: None, 2: None}  # Track last move of each player
selected_piece = None  # Track the currently selected piece
move_piece_mode = False  # Flag to indicate if move piece mode is active
boom_row_mode = False  # Flag to indicate if boom row mode is active
boom_col_mode = False  # Flag to indicate if boom column mode is active
hovered_cell = None  # Track the currently hovered cell
shake_animation = None  # Track shake animation
shake_start_time = None  # Start time for shake animation
shake_duration = 0.5  # Duration of the shake animation in seconds
winner = None  # Track the winner of the game
turn_counter = 0  # Count the number of turns
bot_thinking = False  # Flag to indicate if the bot is thinking

# Pygame screen setup
screen = pygame.display.set_mode((SCREEN_SIZE + BUTTON_WIDTH + 20, SCREEN_SIZE))  
pygame.display.set_caption("Gomoku with Special Abilities")  

# Fonts
font = pygame.font.Font(None, 36)  

# Function to reset the game state
def reset_game():
    """
    Reset all game state variables to their initial values.
    """
    global board, player_turn, moves_left, removes_left, last_move, selected_piece
    global move_piece_mode, boom_row_mode, boom_col_mode, hovered_cell, shake_animation, shake_start_time, winner, turn_counter, boom_used, bot_thinking
    board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # Reset the game board
    player_turn = 1  # Reset to player 1's turn
    moves_left = {1: MOVES_PER_PLAYER, 2: MOVES_PER_PLAYER}  # Reset moves left
    removes_left = {1: REMOVES_PER_PLAYER, 2: REMOVES_PER_PLAYER}  # Reset removes left
    boom_used = {1: False, 2: False}  # Reset boom used flag
    last_move = {1: None, 2: None}  # Reset last move
    selected_piece = None  # Reset selected piece
    move_piece_mode = False  # Reset move piece mode
    boom_row_mode = False  # Reset boom row mode
    boom_col_mode = False  # Reset boom column mode
    hovered_cell = None  # Reset hovered cell
    shake_animation = None  # Reset shake animation
    shake_start_time = None  # Reset shake start time
    winner = None  # Reset winner
    turn_counter = 0  # Reset turn counter
    bot_thinking = False  # Reset bot thinking flag

# Function to print the board state to the console
def print_board():
    """
    Print the current state of the board to the console for debugging.
    """
    print("\nBoard State:")
    for row in board:
        print(" ".join(str(cell) for cell in row))  # Print each row of the board
    print("\n")

# Function to draw the game board
def draw_board():
    """
    Draw the game board, pieces, and any active animations.
    """
    global shake_animation

    screen.fill(WHITE)  # Fill the screen with white color
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            color = WHITE  
            if board[row][col] == 1:
                color = BLACK 
            elif board[row][col] == 2:
                color = RED 

            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)  # Define the cell rectangle
            if shake_animation and (row, col) == shake_animation:
                elapsed = time.time() - shake_start_time
                if elapsed < shake_duration:
                    offset = int(5 * math.sin(elapsed * 20 * math.pi)) 
                    rect.x += offset  # Apply shake effect
                else:
                    shake_animation = None  # End shake animation

            pygame.draw.rect(screen, color, rect) 

            if selected_piece and selected_piece == (row, col):
                pygame.draw.rect(screen, YELLOW, rect, 3)  # Highlight selected piece
            elif hovered_cell:
                if boom_row_mode and row == hovered_cell[0]:
                    pygame.draw.rect(screen, YELLOW, rect, 3)  # Highlight row for boom row mode
                elif boom_col_mode and col == hovered_cell[1]:
                    pygame.draw.rect(screen, YELLOW, rect, 3)  # Highlight column for boom column mode
                else:
                    pygame.draw.rect(screen, BLACK, rect, 1)  # Draw normal cell border
            else:
                pygame.draw.rect(screen, BLACK, rect, 1)  # Draw normal cell border

    draw_buttons()  
    if winner is not None:
        draw_winner_modal() 

# Function to draw the buttons and other UI elements
def draw_buttons():
    """
    Draw the buttons and game information on the right side of the screen.
    """
    move_button = pygame.Rect(SCREEN_SIZE + 10, 50, BUTTON_WIDTH, BUTTON_HEIGHT)  
    boom_row_button = pygame.Rect(SCREEN_SIZE + 10, 100, BUTTON_WIDTH, BUTTON_HEIGHT) 
    boom_col_button = pygame.Rect(SCREEN_SIZE + 10, 150, BUTTON_WIDTH, BUTTON_HEIGHT)  
    moves_left_text = font.render(f'Moves Left: {moves_left[player_turn]}', True, BLACK)  
    removes_left_text = font.render(f'Removes Left: {removes_left[player_turn]}', True, BLACK)  
    turn_text = font.render(f'Turn: {turn_counter}', True, BLACK)  

    # Update button colors based on active mode
    move_button_color = GREEN if move_piece_mode else GRAY  
    boom_row_button_color = GREEN if boom_row_mode else GRAY 
    boom_col_button_color = GREEN if boom_col_mode else GRAY 

    pygame.draw.rect(screen, move_button_color, move_button)  
    pygame.draw.rect(screen, boom_row_button_color, boom_row_button)  
    pygame.draw.rect(screen, boom_col_button_color, boom_col_button)  

    screen.blit(font.render('Move Piece', True, BLACK), (SCREEN_SIZE + 20, 60))  
    screen.blit(font.render('Boom Row', True, BLACK), (SCREEN_SIZE + 20, 110)) 
    screen.blit(font.render('Boom Column', True, BLACK), (SCREEN_SIZE + 20, 160)) 
    screen.blit(moves_left_text, (SCREEN_SIZE + 10, 210))  
    screen.blit(removes_left_text, (SCREEN_SIZE + 10, 260))  
    screen.blit(turn_text, (SCREEN_SIZE + 10, 310)) 

# Function to check if a move is valid
def is_valid_move(row, col, player):
    """
    Check if a move is valid for a given player.
    """
    if board[row][col] != 0:  
        return False
    if last_move[player]:  
        last_row, last_col = last_move[player]
        if row == last_row or col == last_col:
            return False
    return True  

# Function to check if a player has won the game
def check_win(player):
    """
    Check if the given player has won by forming a line of 5 consecutive pieces.
    """
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if (check_line(row, col, 1, 0, player) or
                check_line(row, col, 0, 1, player) or
                check_line(row, col, 1, 1, player) or
                check_line(row, col, 1, -1, player)):
                return True  # Player has won
    return False  # Player has not won

# Function to check a line of 5 cells for a win
def check_line(row, col, d_row, d_col, player):
    """
    Check a specific line (horizontal, vertical, or diagonal) for 5 consecutive pieces of the same player.
    """
    count = 0
    for i in range(5):  # Check up to 5 consecutive positions
        if 0 <= row + i * d_row < GRID_SIZE and 0 <= col + i * d_col < GRID_SIZE:
            if board[row + i * d_row][col + i * d_col] == player:
                count += 1  # Count consecutive pieces of the same player
            else:
                break
    if count == 5:
        return True  
    return False  

# Function to check if a player has won in a given matrix state
def check_win_state(matrix, player):
    """
    Check if the given player has won in the provided matrix state.
    """
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if (check_line_matrix(matrix, row, col, 1, 0, player) or
                check_line_matrix(matrix, row, col, 0, 1, player) or
                check_line_matrix(matrix, row, col, 1, 1, player) or
                check_line_matrix(matrix, row, col, 1, -1, player)):
                return True  # Player has won
    return False  # Player has not won

# Function to check a line of 5 cells for a win in a given matrix state
def check_line_matrix(matrix, row, col, d_row, d_col, player):
    """
    Check a specific line (horizontal, vertical, or diagonal) in the given matrix for 5 consecutive pieces of the same player.
    """
    count = 0
    for i in range(5):  # Check up to 5 consecutive positions
        if 0 <= row + i * d_row < GRID_SIZE and 0 <= col + i * d_col < GRID_SIZE:
            if matrix[row + i * d_row][col + i * d_col] == player:
                count += 1  # Count consecutive pieces of the same player
            else:
                break
    if count == 5:
        return True  
    return False  

# Function to convert the board to a tuple (for caching)
def board_to_tuple(board):
    """
    Convert the board state to a tuple to allow caching in the minimax function.
    """
    return tuple(tuple(row) for row in board)

# Function to get squares to check for possible moves
def get_squares_to_check(matrix):
    """
    Get a list of squares to check for possible moves.
    """
    adjacent = []
    forced_wins = []

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if not matrix[i][j] and is_touching_occupied(matrix, i, j):
                adjacent.append((i, j))  # Add adjacent empty cell

                matrix[i][j] = 1  # Temporarily place player 1's piece
                if check_win_state(matrix, 1):
                    forced_wins.append((i, j))  # Add to forced wins if player 1 wins

                matrix[i][j] = 2  # Temporarily place player 2's piece
                if check_win_state(matrix, 2):
                    forced_wins.append((i, j))  # Add to forced wins if player 2 wins

                matrix[i][j] = 0  # Reset the cell

    return forced_wins if forced_wins else adjacent

# Function to check if a cell is adjacent to an occupied cell
def is_touching_occupied(matrix, i, j):
    """
    Check if a cell is adjacent to an occupied cell.
    """
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),  # Horizontal and vertical directions
        (1, 1), (-1, 1), (-1, -1), (1, -1)  # Diagonal directions
    ]
    for d in directions:
        ni, nj = i + d[0], j + d[1]
        if 0 <= ni < len(matrix) and 0 <= nj < len(matrix[0]) and matrix[ni][nj]:
            return True  
    return False  

# Function to convert a dictionary to a tuple (for caching)
def dict_to_tuple(d):
    """
    Convert a dictionary to a tuple to allow caching in the minimax function.
    """
    return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items())

# Transposition table for caching states
transposition_table = {}

# Function to get neighboring cells within a given radius
def get_neighbors(row, col, radius=1):
    """
    Get a list of neighboring cells within a given radius.
    """
    neighbors = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue  # Skip the cell itself
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
                neighbors.append((new_row, new_col))  # Add valid neighbor
    return neighbors

# Minimax algorithm with alpha-beta pruning and caching
@lru_cache(maxsize=None)
def minimax(matrix_tuple, depth, alpha, beta, is_ai_turn, moves_left_tuple, removes_left_tuple, turn_counter, boom_used_tuple, last_move_tuple):
    """
    Minimax algorithm with alpha-beta pruning to determine the best move for the AI.
    """
    matrix = [list(row) for row in matrix_tuple]
    moves_left = {k: v for k, v in moves_left_tuple}
    removes_left = {k: v for k, v in removes_left_tuple}
    boom_used = {k: v for k, v in boom_used_tuple}
    last_move = {k: v for k, v in last_move_tuple}
    current_player = 2 if is_ai_turn else 1
    opponent = 1 if is_ai_turn else 2

    # Check transposition table
    state = (matrix_tuple, depth, is_ai_turn, dict_to_tuple(moves_left), dict_to_tuple(removes_left), turn_counter, dict_to_tuple(boom_used), dict_to_tuple(last_move))
    if state in transposition_table:
        return transposition_table[state]

    # Base case: maximum depth reached or a player has won
    if depth == 0 or check_win_state(matrix, 1) or check_win_state(matrix, 2):
        return static_evaluation(matrix), None

    best_score = -float('inf') if is_ai_turn else float('inf')
    best_move = None

    # Get a list of squares to check for possible moves
    squares_to_check = get_squares_to_check(matrix)
    squares_to_check = sorted(squares_to_check, key=lambda move: move_score(matrix, move, is_ai_turn), reverse=True)

    for move in squares_to_check:
        y, x = move

        # Skip moves in the same row or column as the last move for the current player
        if last_move[current_player] and (y == last_move[current_player][0] or x == last_move[current_player][1]):
            continue

        matrix[y][x] = current_player
        print(f"Evaluating move at ({y}, {x}) for player {current_player}")

        # Check if the move results in a win
        if check_win_state(matrix, current_player):
            matrix[y][x] = 0
            return (float('inf') if is_ai_turn else -float('inf')), (y, x, "place")

        # Recursively call minimax with the new board state
        score, _ = minimax(
            board_to_tuple(matrix), depth - 1, alpha, beta, not is_ai_turn,
            dict_to_tuple(moves_left), dict_to_tuple(removes_left), turn_counter,
            dict_to_tuple(boom_used), dict_to_tuple(last_move)
        )
        matrix[y][x] = 0

        # Update the best score and move based on the current player's turn
        if is_ai_turn:
            if score > best_score:
                best_score = score
                best_move = (y, x, "place")
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score = score
                best_move = (y, x, "place")
            beta = min(beta, best_score)

        if beta <= alpha:
            break

    # Consider move piece action
    if is_ai_turn and moves_left[2] > 0:
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if matrix[row][col] == 2:
                    neighbors = get_neighbors(row, col)
                    for new_row, new_col in neighbors:
                        if matrix[new_row][new_col] == 0 and is_valid_move(new_row, new_col, 2):
                            matrix[row][col] = 0
                            matrix[new_row][new_col] = 2
                            print(f"Evaluating move piece from ({row}, {col}) to ({new_row}, {new_col}) for AI")

                            # Check if the move results in a win
                            if check_win_state(matrix, current_player):
                                matrix[new_row][new_col] = 0
                                matrix[row][col] = 2
                                return float('inf'), (row, col, "move", new_row, new_col)

                            # Recursively call minimax with the new board state
                            score, _ = minimax(
                                board_to_tuple(matrix), depth - 1, alpha, beta, False,
                                dict_to_tuple({1: moves_left[1], 2: moves_left[2] - 1}),
                                dict_to_tuple(removes_left), turn_counter, dict_to_tuple(boom_used),
                                dict_to_tuple({1: last_move[1], 2: (new_row, new_col)})
                            )
                            matrix[new_row][new_col] = 0
                            matrix[row][col] = 2

                            if score > best_score:
                                best_score = score
                                best_move = (row, col, "move", new_row, new_col)
                            alpha = max(alpha, best_score)

                            if beta <= alpha:
                                break
                    if beta <= alpha:
                        break
            if beta <= alpha:
                break

    # Consider boom row action
    if is_ai_turn and removes_left[2] > 0 and turn_counter >= BOOM_LIMIT_ROUNDS and not boom_used[2]:
        for row in range(GRID_SIZE):
            original_row = matrix[row].copy()
            matrix[row] = [0] * GRID_SIZE
            print(f"Evaluating boom row {row} for AI")

            # Recursively call minimax with the new board state
            score, _ = minimax(
                board_to_tuple(matrix), depth - 1, alpha, beta, False,
                dict_to_tuple(moves_left), dict_to_tuple(removes_left), turn_counter,
                dict_to_tuple({1: boom_used[1], 2: True}), dict_to_tuple(last_move)
            )
            matrix[row] = original_row

            if score > best_score:
                best_score = score
                best_move = (row, 0, "remove_row")
            alpha = max(alpha, best_score)

            if beta <= alpha:
                break

    # Consider boom column action
    if is_ai_turn and removes_left[2] > 0 and turn_counter >= BOOM_LIMIT_ROUNDS and not boom_used[2]:
        for col in range(GRID_SIZE):
            original_col = [matrix[row][col] for row in range(GRID_SIZE)]
            for row in range(GRID_SIZE):
                matrix[row][col] = 0
            print(f"Evaluating boom column {col} for AI")

            # Recursively call minimax with the new board state
            score, _ = minimax(
                board_to_tuple(matrix), depth - 1, alpha, beta, False,
                dict_to_tuple(moves_left), dict_to_tuple(removes_left), turn_counter,
                dict_to_tuple({1: boom_used[1], 2: True}), dict_to_tuple(last_move)
            )
            for row in range(GRID_SIZE):
                matrix[row][col] = original_col[row]

            if score > best_score:
                best_score = score
                best_move = (0, col, "remove_col")
            alpha = max(alpha, best_score)

            if beta <= alpha:
                break

    # Check for no legal moves for opponent
    if best_move and not has_legal_moves(matrix, opponent, moves_left, removes_left, turn_counter):
        if has_legal_moves(matrix, current_player, moves_left, removes_left, turn_counter):
            best_score = float('inf') if is_ai_turn else -float('inf')

    # Store result in transposition table
    transposition_table[state] = (best_score, best_move)

    return best_score, best_move

# Check if a player has any legal moves left
def has_legal_moves(matrix, player, moves_left, removes_left, turn_counter):
    """
    Check if the player has any legal moves left.
    """
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if matrix[row][col] == 0 and is_valid_move(row, col, player):
                return True 
            if matrix[row][col] == player and moves_left[player] > 0:
                for new_row in range(GRID_SIZE):
                    for new_col in range(GRID_SIZE):
                        if matrix[new_row][new_col] == 0 and is_valid_move(new_row, new_col, player):
                            return True
            if removes_left[player] > 0 and turn_counter >= BOOM_LIMIT_ROUNDS:
                return True  
    return False  

# Static evaluation of the board state
def static_evaluation(matrix):
    """
    Evaluate the current state of the board.
    """
    ai_score = 0  
    player_score = 0 

    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if matrix[row][col] == 2:
                ai_score += score_position(matrix, row, col, 2)  # Evaluate AI pieces
            elif matrix[row][col] == 1:
                player_score += score_position(matrix, row, col, 1)  # Evaluate human player pieces

    return ai_score - player_score  

# Score a specific position
def score_position(matrix, row, col, player):
    """
    Calculate the score of a specific position on the board for the given player.
    """
    score = 0
    score += evaluate_line(matrix, row, col, 1, 0, player)  # Horizontal line
    score += evaluate_line(matrix, row, col, 0, 1, player)  # Vertical line
    score += evaluate_line(matrix, row, col, 1, 1, player)  # Diagonal (top-left to bottom-right)
    score += evaluate_line(matrix, row, col, 1, -1, player)  # Anti-diagonal (top-right to bottom-left)
    return score

# Evaluate a line for scoring
def evaluate_line(matrix, row, col, d_row, d_col, player):
    """
    Evaluate a specific line (horizontal, vertical, or diagonal) for scoring.
    """
    count = 0
    for i in range(5): 
        if 0 <= row + i * d_row < len(matrix) and 0 <= col + i * d_col < len(matrix[0]):
            if matrix[row + i * d_row][col + i * d_col] == player:
                count += 1  
            else:
                break
    return count ** 2 

# Calculate score of a move
def move_score(matrix, move, is_ai_turn):
    """
    Calculate the score of a move for the given player.
    """
    y, x = move
    player = 2 if is_ai_turn else 1
    return score_position(matrix, y, x, player)

# Execute the bot's move
def bot_move():
    """
    Execute the AI bot's move using the minimax algorithm.
    """
    global winner, turn_counter, player_turn, bot_thinking, selected_piece
    bot_thinking = True  # Bot starts thinking
    board_copy = copy.deepcopy(board)
    _, move = minimax(board_to_tuple(board_copy), 4, float('-inf'), float('inf'), True, dict_to_tuple(moves_left), dict_to_tuple(removes_left), turn_counter, dict_to_tuple(boom_used), dict_to_tuple(last_move))  # Increased depth to 6
    if move:
        print(f"Bot decision: {move}")
        if move[2] == "place":
            row, col = move[0], move[1]
            board[row][col] = 2
            last_move[2] = (row, col)
        elif move[2] == "move":
            row, col, new_row, new_col = move[0], move[1], move[3], move[4]
            print(f"Bot moving piece from ({row}, {col}) to ({new_row}, {new_col})")
            # Perform the move in the board
            board[row][col] = 0
            board[new_row][new_col] = 2
            moves_left[2] -= 1
            last_move[2] = (new_row, new_col)
        elif move[2] == "remove_row":
            row = move[0]
            for col in range(GRID_SIZE):
                board[row][col] = 0
            removes_left[2] -= 1
            boom_used[2] = True
        elif move[2] == "remove_col":
            col = move[1]
            for row in range(GRID_SIZE):
                board[row][col] = 0
            removes_left[2] -= 1
            boom_used[2] = True

        print_board()

        # Check for a win after the bot's move
        if check_win(2):
            winner = 2
        else:
            player_turn = 1

    bot_thinking = False  # Bot finishes thinking

# Handle piece movement
def handle_move_piece(row, col):
    """
    Handle the logic for moving a piece on the board.
    """
    global selected_piece, move_piece_mode, winner
    if selected_piece:
        prev_row, prev_col = selected_piece
        if board[row][col] == 0:
            print(f"Moving piece from ({prev_row}, {prev_col}) to ({row}, {col})")
            board[row][col] = board[prev_row][prev_col]
            board[prev_row][prev_col] = 0
            selected_piece = None
            move_piece_mode = False
            moves_left[player_turn] -= 1
            last_move[player_turn] = (row, col)  # Update the last move to the new position
            print_board()
            if check_win(player_turn):
                winner = player_turn
            else:
                switch_turn()
        else:
            selected_piece = None
            move_piece_mode = False
    elif board[row][col] == player_turn:
        selected_piece = (row, col)

# Handle boom action
def handle_boom(row, col, axis):
    """
    Handle the logic for the boom action (removing a row or column).
    """
    global boom_row_mode, boom_col_mode, turn_counter, winner, last_move
    if turn_counter < BOOM_LIMIT_ROUNDS:
        return  # Cannot use boom ability before the limit rounds
    if axis == 'row':
        for c in range(GRID_SIZE):
            board[row][c] = 0
        # Reset last move if it was in the removed row
        for player in last_move:
            if last_move[player] and last_move[player][0] == row:
                last_move[player] = None
    else:
        for r in range(GRID_SIZE):
            board[r][col] = 0
        # Reset last move if it was in the removed column
        for player in last_move:
            if last_move[player] and last_move[player][1] == col:
                last_move[player] = None
    boom_row_mode = False
    boom_col_mode = False
    removes_left[player_turn] -= 1
    boom_used[player_turn] = True
    print_board()
    if check_win(player_turn):
        winner = player_turn
    else:
        switch_turn()

# Switch turn to the other player
def switch_turn():
    """
    Switch the turn to the other player.
    """
    global player_turn, turn_counter, winner
    player_turn = 2 if player_turn == 1 else 1
    turn_counter += 1

    # Check if the new player has any legal moves left
    if not has_legal_moves(board, player_turn, moves_left, removes_left, turn_counter):
        if not has_legal_moves(board, 3 - player_turn, moves_left, removes_left, turn_counter):
            winner = 0  # Indicate a draw
        else:
            winner = 3 - player_turn  # The other player wins

    # If it's the bot's turn, make the bot move
    if player_turn == 2 and winner is None:
        threading.Thread(target=bot_move).start()

# Apply shake animation to a cell
def shake_cell(row, col):
    """
    Apply a shake animation to a cell (used for invalid moves).
    """
    global shake_animation, shake_start_time
    shake_animation = (row, col)
    shake_start_time = time.time()

# Draw the winner modal
def draw_winner_modal():
    """
    Draw a modal to display the winner or a draw message.
    """
    global winner
    modal_rect = pygame.Rect((SCREEN_SIZE - MODAL_WIDTH) // 2, (SCREEN_SIZE - MODAL_HEIGHT) // 2, MODAL_WIDTH, MODAL_HEIGHT)
    pygame.draw.rect(screen, WHITE, modal_rect)
    pygame.draw.rect(screen, BLACK, modal_rect, 2)

    if winner == 0:
        winner_text = "It's a draw!"
    else:
        winner_text = f"Player {winner} wins!" if winner else "It's a tie!"
    text_surface = font.render(winner_text, True, BLACK)
    screen.blit(text_surface, ((SCREEN_SIZE - text_surface.get_width()) // 2, (SCREEN_SIZE - MODAL_HEIGHT) // 2 + 40))

    reset_button = pygame.Rect((SCREEN_SIZE - BUTTON_WIDTH) // 2, (SCREEN_SIZE + MODAL_HEIGHT) // 2 - BUTTON_HEIGHT - 20, BUTTON_WIDTH, BUTTON_HEIGHT)
    pygame.draw.rect(screen, GREEN, reset_button)
    pygame.draw.rect(screen, BLACK, reset_button, 2)
    screen.blit(font.render('Reset Game', True, BLACK), ((SCREEN_SIZE - BUTTON_WIDTH) // 2 + 20, (SCREEN_SIZE + MODAL_HEIGHT) // 2 - BUTTON_HEIGHT - 10))

# Main game loop
running = True
while running:
    draw_board()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if bot_thinking:
                continue  # Skip processing input if the bot is thinking
            x, y = pygame.mouse.get_pos()
            if winner is not None:
                reset_button = pygame.Rect((SCREEN_SIZE - BUTTON_WIDTH) // 2, (SCREEN_SIZE + MODAL_HEIGHT) // 2 - BUTTON_HEIGHT - 20, BUTTON_WIDTH, BUTTON_HEIGHT)
                if reset_button.collidepoint(x, y):
                    reset_game()
            elif SCREEN_SIZE < x < SCREEN_SIZE + BUTTON_WIDTH:
                if 50 <= y <= 50 + BUTTON_HEIGHT:
                    if moves_left[player_turn] > 0:
                        move_piece_mode = True
                        boom_row_mode = False
                        boom_col_mode = False
                elif 100 <= y <= 100 + BUTTON_HEIGHT:
                    if removes_left[player_turn] > 0 and turn_counter >= BOOM_LIMIT_ROUNDS:
                        boom_row_mode = True
                        boom_col_mode = False
                        move_piece_mode = False
                elif 150 <= y <= 150 + BUTTON_HEIGHT:
                    if removes_left[player_turn] > 0 and turn_counter >= BOOM_LIMIT_ROUNDS:
                        boom_col_mode = True
                        boom_row_mode = False
                        move_piece_mode = False
            else:
                row, col = y // CELL_SIZE, x // CELL_SIZE
                if move_piece_mode:
                    handle_move_piece(row, col)
                elif boom_row_mode:  # Ensure correct handling of boom row
                    handle_boom(row, col, 'row')
                elif boom_col_mode:  # Ensure correct handling of boom column
                    handle_boom(row, col, 'col')
                elif is_valid_move(row, col, player_turn):
                    board[row][col] = player_turn
                    last_move[player_turn] = (row, col)
                    print_board()
                    if check_win(player_turn):
                        winner = player_turn
                    else:
                        switch_turn()
                else:
                    shake_cell(row, col)
        elif event.type == pygame.MOUSEMOTION:
            if bot_thinking:
                continue  # Skip processing input if the bot is thinking
            x, y = pygame.mouse.get_pos()
            row, col = y // CELL_SIZE, x // CELL_SIZE
            hovered_cell = (row, col) if (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE) else None

    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
