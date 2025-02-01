import pygame
import sys
import time
import jax.numpy as jnp
from jax import jit, vmap

pygame.init()

# Screen settings
WIDTH, HEIGHT = 640, 640
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
pygame.display.set_caption("2D Collision Simulation")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

FPS = 60
timing = True
start_time_collisions = time.time()

# Convert Cartesian (x, y) to Pygame coordinates
def to_pygame_coords(pos):
    return int(WIDTH // 2 + float(pos[0])), int(HEIGHT // 2 - float(pos[1]))

# Bodies stored as JAX arrays
bodies_data = {
    "mass": jnp.array([1.0] * 10),
    "position": jnp.array([
        [-30, 0], [25, 0], [30, 0], [75, -13], [50, 20], 
        [10, 40], [60, -30], [-40, 50], [20, -20], [0, -10]
    ]),
    "velocity": jnp.array([
        [0, 0], [10, -4], [-15, 0], [0, 0], [3, -2], 
        [5, 5], [-10, 8], [2, -1], [6, -3], [-4, 7]
    ]),
    "radius": jnp.array([10, 5, 10, 8, 7, 3, 6, 4, 9, 2])
}

waiting_times = []
times_array = []
times_radii = []

@jit
def update_positions(position, velocity):
    return position + velocity / FPS

def main():
    clock = pygame.time.Clock()
    global bodies_data

    start_time = time.time()
    print(f'Target FPS time: {(1/FPS)*1000:.3f} ms')

    while True:
        frame_start = time.time()
        screen.fill(WHITE)  # Clear screen

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # **Optimized Position Update Using JAX**
        bodies_data["position"] = vmap(update_positions)(bodies_data["position"], bodies_data["velocity"])

        update_body_collisions(bodies_data)

        # Draw circles
        for pos, radius in zip(bodies_data["position"], bodies_data["radius"]):
            pygame_coords = to_pygame_coords(pos)
            pygame.draw.circle(screen, RED, pygame_coords, radius)

        pygame.display.flip()  # Update display

        if timing:
            current_time = time.time()
            time_spent = current_time - frame_start  # Time spent in computations
            time_waiting = max(1/FPS - time_spent, 0)  # Avoid negative waiting times

            waiting_times.append(time_waiting)

            # Every second, calculate and print the average waiting time
            if current_time - start_time >= 1:
                avg_waiting_time = sum(waiting_times) / len(waiting_times) * 1000  # Convert to ms
                print(f"Average waiting time over 1s: {avg_waiting_time:.3f} ms")
                print(f"Min waiting time over 1s: {min(waiting_times)*1000:.3f} ms")
                waiting_times.clear()  # Reset the list
                start_time = current_time  # Reset the timer

        clock.tick(FPS)

def create_comparison_arrays(arr):
    """ Generates arrays for element-wise collision checks """
    n = len(arr)
    first = jnp.repeat(arr[:-1], jnp.arange(n-1, 0, -1), axis=0)
    second = jnp.concatenate([arr[i+1:] for i in range(n-1)], axis=0)
    return first, second

@jit
def collision_check(positions_a, positions_b, radii_a, radii_b):
    """ Checks if pairs of bodies have collided """
    a_to_b = positions_b - positions_a
    a_to_b_size = jnp.linalg.norm(a_to_b, axis=1)
    return a_to_b_size < (radii_a + radii_b)

def update_body_collisions(bodies):
    global start_time_collisions

    if timing:
        frame_start = time.time()

    positions_a, positions_b = create_comparison_arrays(bodies["position"])

    if timing:
        current_time = time.time()
        times_array.append(current_time - frame_start)
        frame_start = time.time()

    radii_a, radii_b = create_comparison_arrays(bodies["radius"])

    if timing:
        current_time = time.time()
        times_radii.append(current_time - frame_start)

        # Every second, print timing stats
        if current_time - start_time_collisions >= 1:
            avg_array_time = sum(times_array) / len(times_array) * 1000
            avg_radii_time = sum(times_radii) / len(times_radii) * 1000
            print(f"Average array time over 1s: {avg_array_time:.3f} ms")
            print(f"Average radii time over 1s: {avg_radii_time:.3f} ms")
            times_array.clear()
            times_radii.clear()
            start_time_collisions = current_time  # Reset the timer

    # Element-wise collision check
    collision_check(positions_a, positions_b, radii_a, radii_b)

main()
