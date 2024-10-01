import pygame
import random
import os


pygame.font.init()

score_font = pygame.font.SysFont('Arial', 50)

screenWidth = 500
screenHeight = 800

screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption("Complex Flappy Bird AI")

Bird1 = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird1.png')))
Bird2 = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird2.png')))
Bird3 = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird3.png')))

pipeImage = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'pipe.png')))
bgImage = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bg.png')))
baseImage = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'base.png')))

birdImages = [Bird1, Bird2, Bird3]

bgWidth = bgImage.get_width()

scroll_speed = 4

background_x1 = 0
background_x2 = bgWidth

class Bird:
    birdImages = birdImages

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = 0
        self.gravityStrength = 1
        self.jumpStrength = -14
        self.maxVelocity = 10
        self.rotationAngle = 0
        self.targetAngle = 0
        self.image = self.birdImages[1]
        self.mask = pygame.mask.from_surface(self.image)

    def gravity(self):
        # the bird is affected by gravity
        self.velocity += self.gravityStrength

        # maximum descent velocity
        if self.velocity > self.maxVelocity:
            self.velocity = self.maxVelocity

        # maximum ascent velocity
        elif self.velocity < -self.maxVelocity:
            self.velocity = -self.maxVelocity

        self.y += self.velocity

    def jump(self):
        self.velocity = self.jumpStrength
        self.rotationAngle = 30

    def rotation(self):
        if self.velocity > 0:
            self.targetAngle = -30
        if self.velocity < 0:
            self.targetAngle = 30

        rotationSpeed = 5

        if self.rotationAngle < self.targetAngle:
            self.rotationAngle = min(self.rotationAngle + rotationSpeed, self.targetAngle)
        elif self.rotationAngle > self.targetAngle:
            self.rotationAngle = max(self.rotationAngle - rotationSpeed, self.targetAngle)

    def draw(self, screen):
        if self.rotationAngle > 0:
            self.image = birdImages[2]
        elif self.rotationAngle < 0:
            self.image = birdImages[0]
        else:
            self.image = birdImages[1]

        rotatedImage = pygame.transform.rotate(self.image, self.rotationAngle)
        newRectangle = rotatedImage.get_rect(center=self.image.get_rect(topleft=(self.x, self.y)).center)
        screen.blit(rotatedImage, newRectangle.topleft)
        self.mask = pygame.mask.from_surface(rotatedImage)

    def get_rect(self):
        return self.image.get_rect(topleft=(self.x, self.y))



class Pipe:
    pipeImage = pipeImage

    def __init__(self, x, y, random):
        self.x = x
        self.y = y
        self.top = 0
        self.bottom = 0
        self.velocityX = -scroll_speed
        self.velocityY = random
        self.passed = False


        self.topImage = pygame.transform.flip(pipeImage, False, True)
        self.bottomImage = pipeImage

        self.topmask = pygame.mask.from_surface(self.topImage)
        self.botmask = pygame.mask.from_surface(self.bottomImage)

        self.height = self.y
        self.pipeGap = 190

    def move(self):
        self.x += self.velocityX
        self.y += self.velocityY

        self.top = self.y - (self.pipeGap / 2) - self.topImage.get_height()
        self.bottom = self.y + (self.pipeGap / 2)

    def draw(self, screen):
        screen.blit(self.topImage, (self.x, self.top))
        screen.blit(self.bottomImage, (self.x, self.bottom))

    def get_top_rect(self):
        top = self.y - (self.pipeGap / 2) - self.topImage.get_height()
        return self.topImage.get_rect(topleft=(self.x, top))

    def get_bot_rect(self):
        bot = self.y + (self.pipeGap / 2)
        return self.bottomImage.get_rect(topleft=(self.x, bot))


class Game:
    def __init__(self, render=True):

        self.bird = Bird(200, 300)
        self.pipes = []
        self.score = 0
        self.pipeFreq = 110
        self.pipeTimer = 0
        self.render = render
        self.running = True
        self.background_x1 = 0
        self.background_x2 = bgWidth

    def reset(self):
        self.bird = Bird(200,300)
        self.pipes = []
        self.score = 0
        self.pipeFreq = 110
        self.pipeTimer = 0
        self.background_x1 = 0
        self.background_x2 = bgWidth

    def playStep(self, action):
        reward = 0
        self.score += 1

        if action == 1:
            self.bird.jump()

        self.bird.gravity()
        self.bird.rotation()
        self.pipeTimer += 1

        if self.pipeTimer > self.pipeFreq:
            self.pipeTimer = 0
            new_pipe_y = random.randint(300, 500)
            new_pipe_speed = random.uniform(-2, 2)
            self.pipes.append(Pipe(screenWidth, new_pipe_y, new_pipe_speed))

        nearest_pipe = None
        for pipe in self.pipes:
            pipe.move()

            # Check if the pipe has been passed
            if pipe.x + pipe.topImage.get_width() < self.bird.x and not pipe.passed:
                pipe.passed = True
                self.score += 100
                reward += 100  # Base reward for passing pipe

                # Bonus reward for passing through the center of the gap
                gap_center = pipe.y
                distance_from_center = abs(self.bird.y - gap_center)
                max_distance = pipe.pipeGap / 2
                centered_bonus = 50 * (1 - (distance_from_center / max_distance))
                reward += max(centered_bonus, 0)  # Ensure bonus is non-negative

            if pipe.x < 0:
                self.pipes.remove(pipe)

            # Find the nearest pipe ahead of the bird
            if nearest_pipe is None or (pipe.x > self.bird.x and pipe.x < nearest_pipe.x):
                nearest_pipe = pipe

        # Add distance-based and vertical positioning rewards
        if nearest_pipe:
            distance_to_pipe = nearest_pipe.x - self.bird.x
            normalized_distance = 1 - (distance_to_pipe / screenWidth)
            reward += normalized_distance * 0.1

            gap_center = nearest_pipe.y
            vertical_distance = abs(self.bird.y - gap_center)
            max_vertical_distance = screenHeight / 2
            vertical_positioning = 1 - (vertical_distance / max_vertical_distance)
            reward += vertical_positioning * 0.1

        # Check for collisions or out-of-bounds
        done = False
        if checkCollision(self.bird, self.pipes):
            done = True
            reward = -100 * (1 - self.score / 10000)  # Adjust penalty based on score
        elif self.bird.y > screenHeight or self.bird.y < 0:
            done = True
            reward = -100 * (1 - self.score / 10000)  # Adjust penalty based on score

        reward += 0.01  # Small reward for surviving

        if self.render:
            self.draw()

        return reward, done, self.score

    def draw(self):
        self.background_x1 -= scroll_speed
        self.background_x2 -= scroll_speed

        if self.background_x1 <= -bgWidth:
            self.background_x1 = bgWidth
        if self.background_x2 <= -bgWidth:
            self.background_x2 = bgWidth

        screen.blit(bgImage, (self.background_x1, 0))
        screen.blit(bgImage, (self.background_x2, 0))

        for pipe in self.pipes:
            pipe.draw(screen)

        self.bird.draw(screen)

        score_text = score_font.render(f'Score: {self.score}', True, (255, 255, 255))
        screen.blit(score_text, (screenWidth - score_text.get_width() - 10, 10))

        pygame.display.flip()

        pygame.time.Clock().tick(60)




def checkCollision(bird, pipes):
    bird_rect = bird.get_rect()
    for pipe in pipes:
        toprect = pipe.get_top_rect()
        botrect = pipe.get_bot_rect()
        top_offset = (toprect.x - bird_rect.x, toprect.y - bird_rect.y)
        bottom_offset = (botrect.x - bird_rect.x, botrect.y - bird_rect.y)

        if bird.mask.overlap(pipe.topmask, top_offset) or bird.mask.overlap(pipe.botmask, bottom_offset):
            return True
    return False

def main():
    game = Game()
    while game.running:
        game.draw()
        reward, done, score = game.playStep(0)

        if done:
            game.reset()
            print(f"Game Over! Score: {score}")

if __name__ == "__main__":
    main()


