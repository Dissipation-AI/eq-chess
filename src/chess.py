"""
Chess board encoding and dataset generation utilities for equilibrium propagation training.

This module provides functions to:
1. Encode chess board states as one-hot tensors
2. Generate random chess positions
3. Create game sequences with random legal moves
4. Prepare PyTorch datasets for training
"""

import chess
import torch
import random
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, random_split


# Piece type to one-hot index mapping (using python-chess constants)
# chess.PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6
# We map these to indices 0-5 for one-hot encoding
PIECE_TYPE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def encode_onehot_gamestate(board: chess.Board) -> torch.Tensor:
    """
    Encode a chess board state as a one-hot tensor representation.

    Args:
        board: A chess.Board object representing the current game state

    Returns:
        A tensor of shape (8, 8, 12) where:
        - Channels 0-5: One-hot encoding for white pieces (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
        - Channels 6-11: One-hot encoding for black pieces (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
        - Each square has at most one piece, encoded in the appropriate channel

    Raises:
        ValueError: If the board is not in a valid chess state
    """
    # Validate board state
    if not board.is_valid():
        raise ValueError("Board is not in a valid chess state")

    # Initialize empty tensor: 8 rows x 8 columns x 12 channels
    encoding = torch.zeros((8, 8, 12), dtype=torch.float32)

    # Iterate through all squares on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is not None:
            # Get row and column (rank and file)
            rank = chess.square_rank(square)  # 0-7 (bottom to top)
            file = chess.square_file(square)  # 0-7 (left to right)

            # Get piece type index (0-5)
            piece_index = PIECE_TYPE_TO_INDEX[piece.piece_type]

            # Determine channel offset based on color
            # White pieces: channels 0-5
            # Black pieces: channels 6-11
            channel_offset = 0 if piece.color == chess.WHITE else 6
            channel = channel_offset + piece_index

            # Set the one-hot encoding
            encoding[rank, file, channel] = 1.0

    return encoding


def generate_random_board_states(n: int, max_moves: int = 20) -> List[chess.Board]:
    """
    Generate n random chess board states by playing random legal moves from the starting position.

    Args:
        n: Number of random board states to generate
        max_moves: Maximum number of random moves to play from start position (default: 20)

    Returns:
        List of chess.Board objects representing random valid game states
    """
    boards = []

    for _ in range(n):
        board = chess.Board()

        # Play a random number of moves (between 0 and max_moves)
        num_moves = random.randint(0, max_moves)

        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)

            # If no legal moves or game over, stop
            if not legal_moves or board.is_game_over():
                break

            # Make a random legal move
            move = random.choice(legal_moves)
            board.push(move)

        boards.append(board.copy())

    return boards


def split_dataset(
    boards: List[chess.Board],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[chess.Board], List[chess.Board], List[chess.Board]]:
    """
    Split a list of chess boards into training, validation, and test sets.

    Args:
        boards: List of chess.Board objects
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)

    Returns:
        Tuple of (train_boards, val_boards, test_boards)

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    total = len(boards)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    # Shuffle boards
    shuffled = boards.copy()
    random.shuffle(shuffled)

    # Split
    train_boards = shuffled[:train_size]
    val_boards = shuffled[train_size:train_size + val_size]
    test_boards = shuffled[train_size + val_size:]

    return train_boards, val_boards, test_boards


def generate_game_sequence(
    start_board: Optional[chess.Board] = None,
    num_half_moves: int = 40
) -> List[chess.Board]:
    """
    Generate a sequence of game states by making random legal moves.

    Args:
        start_board: Starting chess position (default: standard starting position)
        num_half_moves: Number of half-moves (plies) to generate (default: 40, which is 20 full moves)

    Returns:
        List of chess.Board objects representing the game sequence.
        The list includes the starting position and all subsequent positions.
        If the game ends before num_half_moves, the sequence stops early.
    """
    if start_board is None:
        board = chess.Board()
    else:
        board = start_board.copy()

    sequence = [board.copy()]

    for _ in range(num_half_moves):
        legal_moves = list(board.legal_moves)

        # Stop if game is over or no legal moves
        if not legal_moves or board.is_game_over():
            break

        # Make a random legal move
        move = random.choice(legal_moves)
        board.push(move)

        # Add current state to sequence
        sequence.append(board.copy())

    return sequence


def generate_multiple_game_sequences(
    num_games: int,
    num_half_moves_per_game: int = 40,
    start_boards: Optional[List[chess.Board]] = None
) -> List[List[chess.Board]]:
    """
    Generate multiple game sequences.

    Args:
        num_games: Number of game sequences to generate
        num_half_moves_per_game: Number of half-moves per game sequence
        start_boards: Optional list of starting positions (must match num_games if provided)

    Returns:
        List of game sequences, where each sequence is a list of chess.Board objects
    """
    if start_boards is not None and len(start_boards) != num_games:
        raise ValueError("Number of start_boards must match num_games")

    sequences = []

    for i in range(num_games):
        start_board = start_boards[i] if start_boards else None
        sequence = generate_game_sequence(start_board, num_half_moves_per_game)
        sequences.append(sequence)

    return sequences


class ChessBoardDataset(Dataset):
    """
    PyTorch Dataset for chess board states.
    """

    def __init__(self, boards: List[chess.Board]):
        """
        Initialize the dataset with a list of chess boards.

        Args:
            boards: List of chess.Board objects
        """
        self.boards = boards

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get the one-hot encoded board state at index idx.

        Args:
            idx: Index of the board to retrieve

        Returns:
            Tensor of shape (8, 8, 12) representing the board state
        """
        return encode_onehot_gamestate(self.boards[idx])


class ChessSequenceDataset(Dataset):
    """
    PyTorch Dataset for sequences of chess board states (for time-series training).
    """

    def __init__(self, sequences: List[List[chess.Board]]):
        """
        Initialize the dataset with a list of game sequences.

        Args:
            sequences: List of game sequences, where each sequence is a list of chess.Board objects
        """
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get the one-hot encoded sequence at index idx.

        Args:
            idx: Index of the sequence to retrieve

        Returns:
            Tensor of shape (sequence_length, 8, 8, 12) representing the game sequence
        """
        sequence = self.sequences[idx]
        encoded_sequence = [encode_onehot_gamestate(board) for board in sequence]
        return torch.stack(encoded_sequence)


def create_random_board_dataset(
    num_boards: int,
    max_moves: int = 20,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[ChessBoardDataset, ChessBoardDataset, ChessBoardDataset]:
    """
    Create train/val/test ChessBoardDataset splits from randomly generated positions.

    Args:
        num_boards: Total number of random board states to generate
        max_moves: Maximum random moves from starting position
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    boards = generate_random_board_states(num_boards, max_moves)
    train_boards, val_boards, test_boards = split_dataset(boards, train_ratio, val_ratio, test_ratio)

    return (
        ChessBoardDataset(train_boards),
        ChessBoardDataset(val_boards),
        ChessBoardDataset(test_boards)
    )


def create_game_sequence_dataset(
    num_games: int,
    num_half_moves_per_game: int = 40,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[ChessSequenceDataset, ChessSequenceDataset, ChessSequenceDataset]:
    """
    Create train/val/test ChessSequenceDataset splits from randomly generated game sequences.

    Args:
        num_games: Number of game sequences to generate
        num_half_moves_per_game: Number of half-moves per game
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    sequences = generate_multiple_game_sequences(num_games, num_half_moves_per_game)

    # Split sequences
    total = len(sequences)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    shuffled = sequences.copy()
    random.shuffle(shuffled)

    train_sequences = shuffled[:train_size]
    val_sequences = shuffled[train_size:train_size + val_size]
    test_sequences = shuffled[train_size + val_size:]

    return (
        ChessSequenceDataset(train_sequences),
        ChessSequenceDataset(val_sequences),
        ChessSequenceDataset(test_sequences)
    )
