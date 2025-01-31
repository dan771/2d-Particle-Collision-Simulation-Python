import pygame

pygame.init()

screen = pygame.display.set_mode((640, 640))

def main():
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

main()