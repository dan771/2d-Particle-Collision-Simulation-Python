import pygame
import sys
from dataclasses import dataclass

pygame.init()

# Screen settings
WIDTH, HEIGHT = 640, 640
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Collision simulation")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Convert Cartesian (x, y) to Pygame coordinates
def to_pygame_coords(pos):
    return int(WIDTH // 2 + pos[0]), int(HEIGHT // 2 - pos[1])

@dataclass
class body:
    mass: float
    position: tuple
    radius: float = 5
    velocity: tuple = (0, 0)


bodies = []
bodies.append(body(mass=1.0, position=(-25, 0), radius=10))
bodies.append(body(mass=1.0, position=(25,0)))
bodies.append(body(mass=1.0, position=(-15, 20), radius=7))
bodies.append(body(mass=1.0, position=(75,-13)))


def main():
    clock = pygame.time.Clock()

    while True:
        screen.fill(WHITE)  # Clear screen

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Draw circles at specified points
        for obj in bodies:
            pygame_coords = to_pygame_coords(obj.position)
            pygame.draw.circle(screen, RED, pygame_coords, obj.radius)

        pygame.display.flip()  # Update display
        clock.tick(60)  # Limit to 60 FPS
        
main()