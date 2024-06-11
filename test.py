import pygame

# Initialize Pygame
pygame.init()

# Load the images
bullet_image = pygame.image.load('image/bullet_left.png')
tank_image = pygame.image.load('image/enemy_1_0.png')

# Get the size of the images
bullet_size = bullet_image.get_size()
tank_size = tank_image.get_size()

print("Bullet size:", bullet_size)
print("Tank size:", tank_size)
