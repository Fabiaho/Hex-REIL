import pygame
import numpy as np
from HexViewer import HexViewer  # Make sure hexviewer.py is in the same directory


def main():
    rows, cols = 5, 5  # Set the size of the board
    # Example board configuration based on the size
    board = [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, -1, -1, -1],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ]

    viewer = HexViewer(800, 600, rows, cols)
    viewer.render(board)
    pygame.time.wait(3000)
    viewer.close()


if __name__ == "__main__":
    main()
