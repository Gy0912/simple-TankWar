import pygame
import sys
import traceback
import wall
import myTank
import enemyTank
import food
import numpy as np
import random
from multiprocessing import Process, Queue
import os

# Q-learning parameters
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 1
EXPLORATION_DECAY = 0.6
MIN_EXPLORATION_RATE = 0.01
DO_WHILE_COUNTER = 0
STATE_SPACE = (630 // 24, 630 // 24, 630 // 24, 630 // 24, 2, 2)  # Add two extra dimensions for the boolean flags
ACTION_SPACE = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SHOOT']

# Initialize Q-table
q_table = np.zeros((STATE_SPACE[0], STATE_SPACE[1], STATE_SPACE[2], STATE_SPACE[3], STATE_SPACE[4], STATE_SPACE[5], len(ACTION_SPACE)))
count = 0

def check_direction(tank, enemies, bullets):
    """Check if there are enemies or bullets in the direction the tank is facing."""
    dx, dy = tank.dir_x, tank.dir_y

    has_enemy = any(e.rect.left == tank.rect.left + dx * 24 and e.rect.top == tank.rect.top + dy * 24 for e in enemies)
    has_bullet = any(b.rect.left == tank.rect.left + dx * 24 and b.rect.top == tank.rect.top + dy * 24 for b in bullets)

    return has_enemy, has_bullet

def get_state(tank, enemies, bullets):
    """Discretize the position of the tank and the closest enemy to get the state."""
    tank_pos = (tank.rect.left // 24, tank.rect.top // 24)
    closest_enemy = min(enemies, key=lambda e: (e.rect.left - tank.rect.left)**2 + (e.rect.top - tank.rect.top)**2)
    enemy_pos = (closest_enemy.rect.left // 24, closest_enemy.rect.top // 24)
    has_enemy, has_bullet = check_direction(tank, enemies, bullets)

    return tank_pos + enemy_pos + (int(has_enemy), int(has_bullet))

def choose_action(state):
    """Choose an action based on the exploration-exploitation trade-off."""
    global EXPLORATION_RATE
    if random.uniform(0, 1) < EXPLORATION_RATE:
        return random.choice(ACTION_SPACE)
    else:
        q_values = q_table[state[0], state[1], state[2], state[3], state[4], state[5]]
        if np.all(q_values == q_values[0]):
            EXPLORATION_RATE = 1
            return random.choice(ACTION_SPACE)
        return ACTION_SPACE[np.argmax(q_values)]

def update_q_table(state, action, reward, next_state):
    """Update Q-values based on the agent's experience."""
    action_index = ACTION_SPACE.index(action)
    next_max = np.max(q_table[next_state[0], next_state[1], next_state[2], next_state[3], next_state[4], next_state[5]])
    q_value = q_table[state[0], state[1], state[2], state[3], state[4], state[5], action_index]
    new_q_value = (1 - LEARNING_RATE) * q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
    q_table[state[0], state[1], state[2], state[3], state[4], state[5], action_index] = new_q_value

def get_reward(tank, enemies, bullets, action, has_enemy, has_bullet):
    """Define the reward function based on the next state."""
    # Reward based on the inverse of the distance to the closest enemy
    closest_enemy = min(enemies, key=lambda e: (e.rect.left - tank.rect.left)**2 + (e.rect.top - tank.rect.top)**2)
    distance_to_enemy = np.sqrt((closest_enemy.rect.left - tank.rect.left)**2 + (closest_enemy.rect.top - tank.rect.top)**2)
    reward = 1 / (distance_to_enemy + 1)  # Add 1 to avoid division by zero

    # Reward for shooting in the direction of bullets or enemies
    if action == 'SHOOT' and (has_enemy or has_bullet):
        reward += 5  # Positive reward for shooting in the direction of a bullet or enemy

    # Penalty for being hit by an enemy bullet
    for bullet in bullets:
        if pygame.sprite.collide_rect(tank, bullet):
            reward -= 10  # Negative reward for being hit by an enemy bullet

    return reward

def write_score_to_file(score, filename):
    """Write the score to a text file."""
    with open(filename, "a") as file:
        file.write(f"Score: {score}\n")

def initialize_game():
    global allTankGroup, mytankGroup, allEnemyGroup, redEnemyGroup, greenEnemyGroup, otherEnemyGroup, enemyBulletGroup
    global myTank_T1, prop, homeSurvive, enemyNumber, score, delay, switch_R1_R2_image, enemyCouldMove

    # Define sprite groups
    allTankGroup = pygame.sprite.Group()
    mytankGroup = pygame.sprite.Group()
    allEnemyGroup = pygame.sprite.Group()
    redEnemyGroup = pygame.sprite.Group()
    greenEnemyGroup = pygame.sprite.Group()
    otherEnemyGroup = pygame.sprite.Group()
    enemyBulletGroup = pygame.sprite.Group()
    
    # Create map
    bgMap = wall.Map()
    
    # Create food/props
    prop = food.Food()
    
    # Create my tank
    myTank_T1 = myTank.MyTank(1)
    allTankGroup.add(myTank_T1)
    mytankGroup.add(myTank_T1)
    
    # Create enemy tanks
    for i in range(1, 4):
        enemy = enemyTank.EnemyTank(i)
        allTankGroup.add(enemy)
        allEnemyGroup.add(enemy)
        if enemy.isred:
            redEnemyGroup.add(enemy)
            continue
        if enemy.kind == 3:
            greenEnemyGroup.add(enemy)
            continue
        otherEnemyGroup.add(enemy)
    
    delay = 100
    enemyNumber = 3
    enemyCouldMove = True
    switch_R1_R2_image = True
    homeSurvive = True
    score = 0  # Initialize score
    return bgMap

def train_tank(queue, q_table):
    global EXPLORATION_RATE, DO_WHILE_COUNTER, score, delay, switch_R1_R2_image, enemyCouldMove, enemyNumber
    pygame.init()
    pygame.mixer.init()
    
    resolution = 630, 630
    screen = pygame.display.set_mode(resolution)
    pygame.display.set_caption(f"Tank War {os.getpid()}")
    
    # Load images, sounds, etc.
    background_image = pygame.image.load(r"image/background.png")
    home_image = pygame.image.load(r"image/home.png")
    home_destroyed_image = pygame.image.load(r"image/home_destroyed.png")
    
    bang_sound = pygame.mixer.Sound(r"music/bang.wav")
    bang_sound.set_volume(1)
    fire_sound = pygame.mixer.Sound(r"music/Gunfire.wav")
    start_sound = pygame.mixer.Sound(r"music/start.wav")
    start_sound.play()
    
    # Enemy tank appearance animation
    appearance_image = pygame.image.load(r"image/appear.png").convert_alpha()
    appearance = [appearance_image.subsurface((i * 48, 0), (48, 48)) for i in range(3)]
    
    # Custom events
    DELAYEVENT = pygame.constants.USEREVENT
    pygame.time.set_timer(DELAYEVENT, 200)
    ENEMYBULLETNOTCOOLINGEVENT = pygame.constants.USEREVENT + 1
    pygame.time.set_timer(ENEMYBULLETNOTCOOLINGEVENT, 1000)
    MYBULLETNOTCOOLINGEVENT = pygame.constants.USEREVENT + 2
    pygame.time.set_timer(MYBULLETNOTCOOLINGEVENT, 200)
    NOTMOVEEVENT = pygame.constants.USEREVENT + 3
    pygame.time.set_timer(NOTMOVEEVENT, 8000)
    
    clock = pygame.time.Clock()
    filename = f"scores_{os.getpid()}.txt"

    bgMap = initialize_game()

    action = "UP"

    while True:
        state = get_state(myTank_T1, allEnemyGroup, enemyBulletGroup)
        # Decay exploration rate
        EXPLORATION_RATE = max(MIN_EXPLORATION_RATE, EXPLORATION_RATE * EXPLORATION_DECAY)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # My tank bullet cooling event
            if event.type == MYBULLETNOTCOOLINGEVENT:
                myTank_T1.bulletNotCooling = True
                
            # Enemy bullet cooling event
            if event.type == ENEMYBULLETNOTCOOLINGEVENT:
                for each in allEnemyGroup:
                    each.bulletNotCooling = True
            
            # Enemy tank stationary event
            if event.type == NOTMOVEEVENT:
                enemyCouldMove = True
            
            # Delay creating enemy tanks
            if event.type == DELAYEVENT:
                if enemyNumber < 4:
                    enemy = enemyTank.EnemyTank()
                    if pygame.sprite.spritecollide(enemy, allTankGroup, False, None):
                        break
                    allEnemyGroup.add(enemy)
                    allTankGroup.add(enemy)
                    enemyNumber += 1
                    if enemy.isred:
                        redEnemyGroup.add(enemy)
                    elif enemy.kind == 3:
                        greenEnemyGroup.add(enemy)
                    else:
                        otherEnemyGroup.add(enemy)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c and pygame.KMOD_CTRL:
                    pygame.quit()
                    sys.exit()
                
                if event.key == pygame.K_e:
                    myTank_T1.levelUp()
                if event.key == pygame.K_q:
                    myTank_T1.levelDown()
                if event.key == pygame.K_3:
                    myTank_T1.levelUp()
                    myTank_T1.levelUp()
                    myTank_T1.level = 3
                if event.key == pygame.K_2:
                    if myTank_T1.speed == 3:
                        myTank_T1.speed = 24
                    else:
                        myTank_T1.speed = 3
                if event.key == pygame.K_1:
                    for x, y in [(11,23),(12,23),(13,23),(14,23),(11,24),(14,24),(11,25),(14,25)]:
                        bgMap.brick = wall.Brick()
                        bgMap.brick.rect.left, bgMap.brick.rect.top = 3 + x * 24, 3 + y * 24
                        bgMap.brickGroup.add(bgMap.brick)                
                if event.key == pygame.K_4:
                    for x, y in [(11,23),(12,23),(13,23),(14,23),(11,24),(14,24),(11,25),(14,25)]:
                        bgMap.iron = wall.Iron()
                        bgMap.iron.rect.left, bgMap.iron.rect.top = 3 + x * 24, 3 + y * 24
                        bgMap.ironGroup.add(bgMap.iron)

        # Draw background
        screen.blit(background_image, (0, 0))
        # Draw bricks
        for each in bgMap.brickGroup:
            screen.blit(each.image, each.rect)
        # Draw iron
        for each in bgMap.ironGroup:
            screen.blit(each.image, each.rect)
        # Draw home
        if homeSurvive:
            screen.blit(home_image, (3 + 12 * 24, 3 + 24 * 24))
        else:
            screen.blit(home_destroyed_image, (3 + 12 * 24, 3 + 24 * 24))
        # Draw my tank 1
        if not (delay % 5):
            switch_R1_R2_image = not switch_R1_R2_image
        if switch_R1_R2_image:
            screen.blit(myTank_T1.tank_R0, (myTank_T1.rect.left, myTank_T1.rect.top))
        else:
            screen.blit(myTank_T1.tank_R1, (myTank_T1.rect.left, myTank_T1.rect.top))
        # Draw enemy tanks
        for each in allEnemyGroup:
            if each.flash:
                if switch_R1_R2_image:
                    screen.blit(each.tank_R0, (each.rect.left, each.rect.top))
                    if enemyCouldMove:
                        allTankGroup.remove(each)
                        each.move(allTankGroup, bgMap.brickGroup, bgMap.ironGroup)
                        allTankGroup.add(each)
                else:
                    screen.blit(each.tank_R1, (each.rect.left, each.rect.top))
                    if enemyCouldMove:
                        allTankGroup.remove(each)
                        each.move(allTankGroup, bgMap.brickGroup, bgMap.ironGroup)
                        allTankGroup.add(each)
            else:
                if each.times > 0:
                    each.times -= 1
                    if each.times <= 10:
                        screen.blit(appearance[2], (3 + each.x * 12 * 24, 3))
                    elif each.times <= 20:
                        screen.blit(appearance[1], (3 + each.x * 12 * 24, 3))
                    elif each.times <= 30:
                        screen.blit(appearance[0], (3 + each.x * 12 * 24, 3))
                    elif each.times <= 40:
                        screen.blit(appearance[2], (3 + each.x * 12 * 24, 3))
                    elif each.times <= 50:
                        screen.blit(appearance[1], (3 + each.x * 12 * 24, 3))
                    elif each.times <= 60:
                        screen.blit(appearance[0], (3 + each.x * 12 * 24, 3))
                    elif each.times <= 70:
                        screen.blit(appearance[2], (3 + each.x * 12 * 24, 3))
                    elif each.times <= 80:
                        screen.blit(appearance[1], (3 + each.x * 12 * 24, 3))
                    elif each.times <= 90:
                        screen.blit(appearance[0], (3 + each.x * 12 * 24, 3))
                if each.times == 0:
                    each.flash = True

        # Draw my tank 1 bullet
        if myTank_T1.bullet.life:
            myTank_T1.bullet.move()
            screen.blit(myTank_T1.bullet.bullet, myTank_T1.bullet.rect)
            for each in enemyBulletGroup:
                if each.life and pygame.sprite.collide_rect(myTank_T1.bullet, each):
                    myTank_T1.bullet.life = False
                    each.life = False
                    score += 50
                    pygame.sprite.spritecollide(myTank_T1.bullet, enemyBulletGroup, True, None)
            for each in allEnemyGroup:
                if pygame.sprite.collide_rect(myTank_T1.bullet, each):
                    bang_sound.play()
                    enemyNumber -= 1
                    if each.kind == 1:
                        score += 100
                    elif each.kind == 2:
                        score += 200
                    else:
                        score += 400
                    allEnemyGroup.remove(each)
                    allTankGroup.remove(each)
                    myTank_T1.bullet.life = False
                    break
            if pygame.sprite.spritecollide(myTank_T1.bullet, bgMap.brickGroup, True, None):
                myTank_T1.bullet.life = False
                myTank_T1.bullet.rect.left, myTank_T1.bullet.rect.right = 3 + 12 * 24, 3 + 24 * 24
            if myTank_T1.bullet.strong:
                if pygame.sprite.spritecollide(myTank_T1.bullet, bgMap.ironGroup, True, None):
                    myTank_T1.bullet.life = False
                    myTank_T1.bullet.rect.left, myTank_T1.bullet.rect.right = 3 + 12 * 24, 3 + 24 * 24
            else:
                if pygame.sprite.spritecollide(myTank_T1.bullet, bgMap.ironGroup, False, None):
                    myTank_T1.bullet.life = False
                    myTank_T1.bullet.rect.left, myTank_T1.bullet.rect.right = 3 + 12 * 24, 3 + 24 * 24
        
        # Draw enemy bullets
        for each in allEnemyGroup:
            if not each.bullet.life and each.bulletNotCooling and enemyCouldMove:
                enemyBulletGroup.remove(each.bullet)
                each.shoot()
                enemyBulletGroup.add(each.bullet)
                each.bulletNotCooling = False
            if each.flash:
                if each.bullet.life:
                    if enemyCouldMove:
                        each.bullet.move()
                    screen.blit(each.bullet.bullet, each.bullet.rect)
                    if pygame.sprite.collide_rect(each.bullet, myTank_T1):
                        bang_sound.play()
                        myTank_T1.rect.left, myTank_T1.rect.top = 3 + 8 * 24, 3 + 24 * 24 
                        each.bullet.life = False
                        # DO_WHILE_COUNTER = DO_WHILE_COUNTER + 20
                        # score = score + DO_WHILE_COUNTER
                        write_score_to_file(score, filename)  # Write score to file on death
                        score = 0  # Reset score after death
                        for i in range(myTank_T1.level + 1):
                            myTank_T1.levelDown()
                        bgMap = initialize_game()  # Reset the game world after death
                    if pygame.sprite.spritecollide(each.bullet, bgMap.brickGroup, True, None):
                        each.bullet.life = False
                    if each.bullet.strong:
                        if pygame.sprite.spritecollide(each.bullet, bgMap.ironGroup, True, None):
                            each.bullet.life = False
                    else:
                        if pygame.sprite.spritecollide(each.bullet, bgMap.ironGroup, False, None):
                            each.bullet.life = False

        # Draw food/props
        if prop.life:
            screen.blit(prop.image, prop.rect)
            if pygame.sprite.collide_rect(myTank_T1, prop):
                score += 10
                if prop.kind == 1:
                    for each in allEnemyGroup:
                        if pygame.sprite.spritecollide(each, allEnemyGroup, True, None):
                            bang_sound.play()
                            enemyNumber -= 1
                            if each.kind == 1:
                                score += 100
                            elif each.kind == 2:
                                score += 200
                            else:
                                score += 400
                    prop.life = False
                if prop.kind == 2:
                    enemyCouldMove = False
                    prop.life = False
                if prop.kind == 3:
                    myTank_T1.bullet.strong = True
                    prop.life = False
                if prop.kind == 4:
                    for x, y in [(11,23),(12,23),(13,23),(14,23),(11,24),(14,24),(11,25),(14,25)]:
                        bgMap.iron = wall.Iron()
                        bgMap.iron.rect.left, bgMap.iron.rect.top = 3 + x * 24, 3 + y * 24
                        bgMap.ironGroup.add(bgMap.iron)                
                    prop.life = False
                if prop.kind == 5:
                    prop.life = False
                if prop.kind == 6:
                    myTank_T1.levelUp()
                    prop.life = False
                if prop.kind == 7:
                    myTank_T1.life += 1
                    prop.life = False

        next_state = get_state(myTank_T1, allEnemyGroup, enemyBulletGroup)
        has_enemy, has_bullet = state[4], state[5]
        reward = get_reward(myTank_T1, allEnemyGroup, enemyBulletGroup, action, has_enemy, has_bullet)
        update_q_table(state, action, reward, next_state)

        action = choose_action(next_state)
        # Perform action
        if action == 'UP':
            allTankGroup.remove(myTank_T1)
            myTank_T1.moveUp(allTankGroup, bgMap.brickGroup, bgMap.ironGroup)
            allTankGroup.add(myTank_T1)
        elif action == 'DOWN':
            allTankGroup.remove(myTank_T1)
            myTank_T1.moveDown(allTankGroup, bgMap.brickGroup, bgMap.ironGroup)
            allTankGroup.add(myTank_T1)
        elif action == 'LEFT':
            allTankGroup.remove(myTank_T1)
            myTank_T1.moveLeft(allTankGroup, bgMap.brickGroup, bgMap.ironGroup)
            allTankGroup.add(myTank_T1)
        elif action == 'RIGHT':
            allTankGroup.remove(myTank_T1)
            myTank_T1.moveRight(allTankGroup, bgMap.brickGroup, bgMap.ironGroup)
            allTankGroup.add(myTank_T1)
        elif action == 'SHOOT':
            if not myTank_T1.bullet.life and myTank_T1.bulletNotCooling:
                fire_sound.play()
                myTank_T1.shoot()
                myTank_T1.bulletNotCooling = False

        delay -= 1
        if not delay:
            delay = 100
        
        pygame.display.flip()
        clock.tick(960000)

if __name__ == "__main__":
    try:
        processes = []
        num_processes = 1  # Number of parallel processes

        for _ in range(num_processes):
            queue = Queue()
            p = Process(target=train_tank, args=(queue, q_table))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        np.save("q_table.npy", q_table)  # Save Q-table to file

    except SystemExit:
        pass
    except:
        traceback.print_exc()
        pygame.quit()
        input()
