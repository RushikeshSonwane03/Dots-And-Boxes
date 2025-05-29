#!/usr/bin/env python3
"""
Dots and Boxes Game - A Data Structures Demonstration
Features: 2D Arrays, HashSets, Dictionaries, Stacks, Linked Lists, Graphs, and AI
"""

from collections import deque, defaultdict
from typing import Dict, List, Tuple, Set, Optional
import random
import copy

class Move:
    """Represents a single move in the game - part of our Linked List structure"""
    def __init__(self, player: int, dot1: Tuple[int, int], dot2: Tuple[int, int]):
        self.player = player
        self.dot1 = dot1
        self.dot2 = dot2
        self.boxes_claimed = []  # Boxes claimed by this move
        self.next = None
        self.prev = None

class MoveHistory:
    """Linked List implementation for tracking game history"""
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def add_move(self, move: Move):
        """Add a move to the end of the history"""
        if not self.head:
            self.head = self.tail = move
        else:
            self.tail.next = move
            move.prev = self.tail
            self.tail = move
        self.size += 1
    
    def get_last_move(self) -> Optional[Move]:
        """Get the last move without removing it"""
        return self.tail
    
    def remove_last_move(self) -> Optional[Move]:
        """Remove and return the last move"""
        if not self.tail:
            return None
        
        move = self.tail
        if self.tail.prev:
            self.tail = self.tail.prev
            self.tail.next = None
        else:
            self.head = self.tail = None
        
        self.size -= 1
        return move

class GameGraph:
    """Graph representation of the game board for analysis"""
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.adjacency_list = defaultdict(list)
        self._build_graph()
    
    def _build_graph(self):
        """Build adjacency list for all possible connections"""
        for r in range(self.rows):
            for c in range(self.cols):
                current = (r, c)
                # Add horizontal connection
                if c < self.cols - 1:
                    neighbor = (r, c + 1)
                    self.adjacency_list[current].append(neighbor)
                    self.adjacency_list[neighbor].append(current)
                # Add vertical connection
                if r < self.rows - 1:
                    neighbor = (r + 1, c)
                    self.adjacency_list[current].append(neighbor)
                    self.adjacency_list[neighbor].append(current)
    
    def get_neighbors(self, dot: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all possible neighbors of a dot"""
        return self.adjacency_list[dot]

class DotsAndBoxes:
    """Main game class implementing multiple data structures"""
    
    def __init__(self, rows: int = 6, cols: int = 6, num_players: int = 2):
        self.rows = rows  # Number of dots vertically
        self.cols = cols  # Number of dots horizontally
        self.num_players = num_players
        
        # 2D Array/Matrix for board representation
        self.board = [[' ' for _ in range(2 * cols - 1)] for _ in range(2 * rows - 1)]
        
        # HashSet for tracking drawn lines (using frozenset for immutable keys)
        self.lines_drawn: Set[frozenset] = set()
        
        # Dictionary for box ownership
        self.boxes: Dict[Tuple[int, int], int] = {}
        
        # Queue for player turns
        self.player_queue = deque(range(1, num_players + 1))
        self.current_player = self.player_queue[0]
        
        # Stack for undo functionality
        self.undo_stack: List[Dict] = []
        
        # Linked List for move history
        self.move_history = MoveHistory()
        
        # Graph representation
        self.graph = GameGraph(rows, cols)
        
        # HashMap for scoreboard
        self.scores: Dict[int, int] = {i: 0 for i in range(1, num_players + 1)}
        
        self._initialize_board()
    
    def _initialize_board(self):
        """Initialize the board with dots"""
        for r in range(0, 2 * self.rows - 1, 2):
            for c in range(0, 2 * self.cols - 1, 2):
                self.board[r][c] = '.'
    
    def _line_key(self, dot1: Tuple[int, int], dot2: Tuple[int, int]) -> frozenset:
        """Create a hashable key for a line between two dots"""
        return frozenset([dot1, dot2])
    
    def is_valid_move(self, dot1: Tuple[int, int], dot2: Tuple[int, int]) -> bool:
        """Check if a move is valid"""
        # Check if dots are within bounds
        if not (0 <= dot1[0] < self.rows and 0 <= dot1[1] < self.cols):
            return False
        if not (0 <= dot2[0] < self.rows and 0 <= dot2[1] < self.cols):
            return False
        
        # Check if dots are adjacent
        if abs(dot1[0] - dot2[0]) + abs(dot1[1] - dot2[1]) != 1:
            return False
        
        # Check if line already exists
        line_key = self._line_key(dot1, dot2)
        return line_key not in self.lines_drawn
    
    def _get_board_position(self, dot1: Tuple[int, int], dot2: Tuple[int, int]) -> Tuple[int, int]:
        """Get the position on the display board for a line"""
        r1, c1 = dot1
        r2, c2 = dot2
        
        board_r = r1 * 2 + (r2 - r1)
        board_c = c1 * 2 + (c2 - c1)
        
        return (board_r, board_c)
    
    def _check_completed_boxes(self, dot1: Tuple[int, int], dot2: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Check which boxes are completed by drawing this line"""
        completed_boxes = []
        
        # Get potential box positions that could be affected
        potential_boxes = []
        
        if dot1[0] == dot2[0]:  # Horizontal line
            # Check box above and below
            r = dot1[0]
            c = min(dot1[1], dot2[1])
            if r > 0:
                potential_boxes.append((r - 1, c))
            if r < self.rows - 1:
                potential_boxes.append((r, c))
        else:  # Vertical line
            # Check box left and right
            r = min(dot1[0], dot2[0])
            c = dot1[1]
            if c > 0:
                potential_boxes.append((r, c - 1))
            if c < self.cols - 1:
                potential_boxes.append((r, c))
        
        # Check if any potential box is now complete
        for box_r, box_c in potential_boxes:
            if self._is_box_complete(box_r, box_c):
                completed_boxes.append((box_r, box_c))
        
        return completed_boxes
    
    def _is_box_complete(self, box_r: int, box_c: int) -> bool:
        """Check if a specific box is complete"""
        if box_r < 0 or box_r >= self.rows - 1 or box_c < 0 or box_c >= self.cols - 1:
            return False
        
        # Check all four sides of the box
        top_left = (box_r, box_c)
        top_right = (box_r, box_c + 1)
        bottom_left = (box_r + 1, box_c)
        bottom_right = (box_r + 1, box_c + 1)
        
        sides = [
            self._line_key(top_left, top_right),      # Top
            self._line_key(bottom_left, bottom_right),  # Bottom
            self._line_key(top_left, bottom_left),     # Left
            self._line_key(top_right, bottom_right)    # Right
        ]
        
        return all(side in self.lines_drawn for side in sides)
    
    def make_move(self, dot1: Tuple[int, int], dot2: Tuple[int, int]) -> bool:
        """Make a move and return True if player gets another turn"""
        if not self.is_valid_move(dot1, dot2):
            return False
        
        # Save game state for undo
        game_state = {
            'lines_drawn': copy.deepcopy(self.lines_drawn),
            'boxes': copy.deepcopy(self.boxes),
            'scores': copy.deepcopy(self.scores),
            'board': copy.deepcopy(self.board)
        }
        self.undo_stack.append(game_state)
        
        # Draw the line
        line_key = self._line_key(dot1, dot2)
        self.lines_drawn.add(line_key)
        
        # Update display board
        board_pos = self._get_board_position(dot1, dot2)
        if dot1[0] == dot2[0]:  # Horizontal line
            self.board[board_pos[0]][board_pos[1]] = '─'
        else:  # Vertical line
            self.board[board_pos[0]][board_pos[1]] = '│'
        
        # Check for completed boxes
        completed_boxes = self._check_completed_boxes(dot1, dot2)
        
        # Create move object for history
        move = Move(self.current_player, dot1, dot2)
        move.boxes_claimed = completed_boxes
        
        # Claim boxes and update scores
        for box in completed_boxes:
            self.boxes[box] = self.current_player
            self.scores[self.current_player] += 1
            # Update display board with player number
            display_r, display_c = box[0] * 2 + 1, box[1] * 2 + 1
            self.board[display_r][display_c] = str(self.current_player)
        
        # Add move to history
        self.move_history.add_move(move)
        
        # Return True if player gets another turn (scored points)
        return len(completed_boxes) > 0
    
    def next_player(self):
        """Move to the next player"""
        self.player_queue.rotate(-1)
        self.current_player = self.player_queue[0]
    
    def undo_last_move(self) -> bool:
        """Undo the last move using stack"""
        if not self.undo_stack:
            return False
        
        # Restore game state
        game_state = self.undo_stack.pop()
        self.lines_drawn = game_state['lines_drawn']
        self.boxes = game_state['boxes']
        self.scores = game_state['scores']
        self.board = game_state['board']
        
        # Remove from move history
        self.move_history.remove_last_move()
        
        return True
    
    def get_available_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get all available moves"""
        moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                current = (r, c)
                for neighbor in self.graph.get_neighbors(current):
                    if self.is_valid_move(current, neighbor):
                        # Avoid duplicate moves
                        if (current, neighbor) not in moves and (neighbor, current) not in moves:
                            moves.append((current, neighbor))
        return moves
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        total_possible_boxes = (self.rows - 1) * (self.cols - 1)
        return len(self.boxes) == total_possible_boxes
    
    def get_winner(self) -> Optional[int]:
        """Get the winner of the game"""
        if not self.is_game_over():
            return None
        
        max_score = max(self.scores.values())
        winners = [player for player, score in self.scores.items() if score == max_score]
        
        return winners[0] if len(winners) == 1 else None  # None for tie
    
    def display_board(self):
        """Display the current board state"""
        print("\n" + "="*50)
        print(f"Current Player: {self.current_player}")
        print("Scores:", " | ".join([f"Player {p}: {s}" for p, s in self.scores.items()]))
        print("="*50)
        
        for row in self.board:
            print(' '.join(row))
        print()
    
    def display_move_history(self):
        """Display the complete move history using linked list"""
        print("\n--- Move History ---")
        current = self.move_history.head
        move_num = 1
        
        while current:
            boxes_info = f" (claimed {len(current.boxes_claimed)} box{'es' if len(current.boxes_claimed) != 1 else ''})" if current.boxes_claimed else ""
            print(f"{move_num}. Player {current.player}: {current.dot1} -> {current.dot2}{boxes_info}")
            current = current.next
            move_num += 1
        print()

class SimpleAI:
    """Simple AI that uses minimax-like evaluation"""
    
    def __init__(self, game: DotsAndBoxes, player_id: int):
        self.game = game
        self.player_id = player_id
    
    def evaluate_move(self, dot1: Tuple[int, int], dot2: Tuple[int, int]) -> int:
        """Evaluate a move's value"""
        # Create a temporary game state
        temp_game = copy.deepcopy(self.game)
        
        # Make the move
        gets_extra_turn = temp_game.make_move(dot1, dot2)
        
        # Calculate score gain
        score_gain = temp_game.scores[self.player_id] - self.game.scores[self.player_id]
        
        # Bonus for getting extra turn
        extra_turn_bonus = 10 if gets_extra_turn else 0
        
        return score_gain * 100 + extra_turn_bonus
    
    def get_best_move(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get the best move for the AI"""
        available_moves = self.game.get_available_moves()
        
        if not available_moves:
            return None
        
        best_move = None
        best_score = -float('inf')
        
        for move in available_moves:
            score = self.evaluate_move(move[0], move[1])
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move

def main():
    """Main game loop"""
    print("🎮 Welcome to Dots and Boxes!")
    print("This game demonstrates various data structures:")
    print("- 2D Arrays (board), HashSets (lines), Dictionaries (boxes, scores)")
    print("- Stacks (undo), Linked Lists (history), Graphs (connections)")
    print()
    
    # Game setup
    rows = int(input("Enter number of dot rows (default 4): ") or 4)
    cols = int(input("Enter number of dot columns (default 4): ") or 4)
    num_players = int(input("Enter number of players (default 2): ") or 2)
    
    use_ai = input("Include AI player? (y/n, default n): ").lower().startswith('y')
    
    game = DotsAndBoxes(rows, cols, num_players)
    ai_player = SimpleAI(game, num_players) if use_ai else None
    
    print(f"\n🎯 Game started! {rows}x{cols} grid, {(rows-1)*(cols-1)} boxes to claim")
    print("Enter moves as: r1,c1 r2,c2 (e.g., '0,0 0,1' to connect top-left dots)")
    print("Commands: 'undo', 'history', 'moves', 'quit'")
    
    while not game.is_game_over():
        game.display_board()
        
        # AI turn
        if use_ai and game.current_player == num_players:
            print(f"🤖 AI Player {game.current_player} is thinking...")
            move = ai_player.get_best_move()
            if move:
                gets_extra_turn = game.make_move(move[0], move[1])
                print(f"AI plays: {move[0]} -> {move[1]}")
                if not gets_extra_turn:
                    game.next_player()
            continue
        
        # Human turn
        try:
            user_input = input(f"Player {game.current_player}, enter your move: ").strip().lower()
            
            if user_input == 'quit':
                print("Game ended by user.")
                break
            elif user_input == 'undo':
                if game.undo_last_move():
                    print("✅ Last move undone!")
                else:
                    print("❌ No moves to undo!")
                continue
            elif user_input == 'history':
                game.display_move_history()
                continue
            elif user_input == 'moves':
                available = game.get_available_moves()
                print(f"Available moves: {len(available)}")
                for i, move in enumerate(available[:10]):  # Show first 10
                    print(f"  {move[0]} -> {move[1]}")
                if len(available) > 10:
                    print(f"  ... and {len(available) - 10} more")
                continue
            
            # Parse move
            parts = user_input.split()
            if len(parts) != 2:
                print("❌ Invalid format! Use: r1,c1 r2,c2")
                continue
            
            dot1 = tuple(map(int, parts[0].split(',')))
            dot2 = tuple(map(int, parts[1].split(',')))
            
            # Make move
            gets_extra_turn = game.make_move(dot1, dot2)
            
            if gets_extra_turn:
                print("🎉 You completed a box! Go again!")
            else:
                game.next_player()
                
        except (ValueError, IndexError):
            print("❌ Invalid input! Use format: r1,c1 r2,c2 (e.g., '0,0 0,1')")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Game over
    game.display_board()
    winner = game.get_winner()
    
    if winner:
        print(f"🏆 Player {winner} wins with {game.scores[winner]} boxes!")
    else:
        print("🤝 It's a tie!")
    
    print("\n📊 Final Statistics:")
    print(f"Total moves played: {game.move_history.size}")
    print(f"Lines drawn: {len(game.lines_drawn)}")
    print(f"Boxes claimed: {len(game.boxes)}")
    
    # Show final move history
    show_history = input("\nShow complete move history? (y/n): ").lower().startswith('y')
    if show_history:
        game.display_move_history()

if __name__ == "__main__":
    main()