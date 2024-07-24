import math
from dataclasses import dataclass
from typing import List, Tuple

import pygame


@dataclass
class HexagonTile:
    """Hexagon class"""

    radius: float
    position: Tuple[float, float]
    colour: Tuple[int, ...]
    highlight_offset: int = 3
    max_highlight_ticks: int = 15

    def __post_init__(self):
        self.vertices = self.compute_vertices()
        self.highlight_tick = 0

    def update(self):
        """Updates tile highlights"""
        if self.highlight_tick > 0:
            self.highlight_tick -= 1

    def compute_vertices(self) -> List[Tuple[float, float]]:
        """Returns a list of the hexagon's vertices as x, y tuples"""
        x, y = self.position
        return [
            (x + self.radius * math.cos(math.radians(angle)),
             y + self.radius * math.sin(math.radians(angle)))
            for angle in range(30, 390, 60)  # Start at 30 degrees for pointy-topped hexagons
        ]

    def compute_neighbours(self, hexagons: List['HexagonTile']) -> List['HexagonTile']:
        """Returns hexagons whose centres are two minimal radiuses away from self.centre"""
        return [hexagon for hexagon in hexagons if self.is_neighbour(hexagon)]

    def collide_with_point(self, point: Tuple[float, float]) -> bool:
        """Returns True if distance from centre to point is less than horizontal_length"""
        return math.dist(point, self.centre) < self.minimal_radius

    def is_neighbour(self, hexagon: 'HexagonTile') -> bool:
        """Returns True if hexagon centre is approximately
        2 minimal radiuses away from own centre"""
        distance = math.dist(hexagon.centre, self.centre)
        return math.isclose(distance, 2 * self.minimal_radius, rel_tol=0.05)

    def render(self, screen) -> None:
        """Renders the hexagon on the screen"""
        pygame.draw.polygon(screen, self.colour, self.vertices)
        pygame.draw.polygon(screen, (0, 0, 0), self.vertices, 2)

    def render_highlight(self, screen, border_colour) -> None:
        """Draws a border around the hexagon with the specified colour"""
        self.highlight_tick = self.max_highlight_ticks
        pygame.draw.aalines(screen, border_colour, closed=True, points=self.vertices)

    @property
    def centre(self) -> Tuple[float, float]:
        """Centre of the hexagon"""
        return self.position

    @property
    def minimal_radius(self) -> float:
        """Horizontal length of the hexagon"""
        return self.radius * math.cos(math.radians(30))

    @property
    def highlight_colour(self) -> Tuple[int, ...]:
        """Colour of the hexagon tile when rendering highlight"""
        offset = self.highlight_offset * self.highlight_tick
        brighten = lambda x, y: x + y if x + y < 255 else 255
        return tuple(brighten(x, offset) for x in self.colour)


class HexViewer:
    def __init__(self, width, height, rows, cols, cell_size=50, header_height=50):
        pygame.init()
        self.cell_size = cell_size
        self.width = width
        self.height = height + header_height  # Increase height to accommodate the header
        self.rows = rows
        self.cols = cols
        self.header_height = header_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.title_font = pygame.font.Font(None, 48)  # Font for the title

    def draw_board(self, board):
        self.screen.fill((255, 255, 255))  # Set background to white
        radius = self.cell_size // 2

        # Draw the title
        title_text = "Hex Game of DEATH " + str(self.rows)  + " x " + str(self.cols)
        title_surface = self.title_font.render(title_text, True, (0, 0, 0))
        self.screen.blit(title_surface, (self.width // 2 - title_surface.get_width() // 2, 10))

        # Calculate padding to center the board
        x_padding = (self.width - (self.cols * 1.5 * radius + 0.5 * radius)) // 2
        y_padding = (self.height - self.header_height - (self.rows * math.sqrt(3) * radius + 0.5 * radius)) // 2 + self.header_height

        # Draw the hexagon tiles
        for y in range(self.rows):
            for x in range(self.cols):
                position = self.hex_to_pixel(x, y, radius, x_padding, y_padding)
                colour = (255, 255, 255)  # Default color is white
                if board[y][x] == 1:
                    colour = (255, 0, 0)  # Red for player 1
                elif board[y][x] == -1:
                    colour = (0, 0, 255)  # Blue for player -1

                hex_tile = HexagonTile(radius, position, colour)
                hex_tile.render(self.screen)

        # Draw the column labels (A, B, C, ...)
        for col in range(self.cols):
            col_label = chr(ord('A') + col)
            pos_x, pos_y = self.hex_to_pixel(col, 0, radius, x_padding, y_padding)
            text_surface = self.font.render(col_label, True, (0, 0, 0))
            self.screen.blit(text_surface, (pos_x - text_surface.get_width() // 2, pos_y - radius * 2))

        # Draw the row labels (1, 2, 3, ...)
        for row in range(self.rows):
            row_label = str(row + 1)
            pos_x, pos_y = self.hex_to_pixel(self.cols - 1, row, radius, x_padding, y_padding)
            text_surface = self.font.render(row_label, True, (0, 0, 0))
            self.screen.blit(text_surface, (pos_x + radius, pos_y - text_surface.get_height() // 2))

        pygame.display.flip()

    def hex_to_pixel(self, q, r, size, x_padding, y_padding):
        x = size * math.sqrt(3) * (q + 0.5 * r)
        y = size * 3 / 2 * r
        return x + x_padding, y + y_padding

    def render(self, board):
        self.draw_board(board)
        self.clock.tick(4)

    def close(self):
        pygame.quit()