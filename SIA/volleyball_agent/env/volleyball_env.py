"""
A custom implementation of the Slime Volleyball environment.
This is a direct copy of the original SlimeVolleyEnv with minimal changes.
"""

import numpy as np
import gym
from gym import spaces
import cv2
import pygame
import math

RAD2DEG = 57.2957795130823209
DEG2RAD = 0.01745329251994329576

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400

GAME_WIDTH = 800
GAME_HEIGHT = 400
WALL_WIDTH = 10
WALL_HEIGHT = 200
BALL_RADIUS = 10
PLAYER_RADIUS = 30
PLAYER_SPEED_SCALE = 10
PLAYER_DAMPING = 0.85
PLAYER_DENSITY = 1.0
PLAYER_FRICTION = 0.1
BALL_DENSITY = 0.2
BALL_FRICTION = 0.2
BALL_ELASTICITY = 0.98
MAX_BALL_SPEED = 15
GRAVITY = -9.8 * 2  # Increased gravity for more dynamic gameplay
TIMESTEP = 1.0/60.0

class VolleyballEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 60
    }

    def __init__(self):
        # Observation Space: 
        # [player_x, player_y, player_vx, player_vy, ball_x, ball_y, ball_vx, ball_vy, opponent_x, opponent_y, opponent_vx, opponent_vy]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        # Action Space: [left/right, jump]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        self.viewer = None
        self.screen = None
        self.clock = None
        
        # Game state
        self.player_pos = np.array([GAME_WIDTH/4, PLAYER_RADIUS])
        self.player_vel = np.array([0.0, 0.0])
        self.ball_pos = np.array([GAME_WIDTH/2, GAME_HEIGHT/2])
        self.ball_vel = np.array([0.0, 0.0])
        self.opponent_pos = np.array([3*GAME_WIDTH/4, PLAYER_RADIUS])
        self.opponent_vel = np.array([0.0, 0.0])
        
        self.score = 0
        self.opponent_score = 0
        self.done = False
        
    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return [seed]
        
    def reset(self):
        # Reset positions and velocities
        self.player_pos = np.array([GAME_WIDTH/4, PLAYER_RADIUS])
        self.player_vel = np.array([0.0, 0.0])
        self.ball_pos = np.array([GAME_WIDTH/2, GAME_HEIGHT/2])
        self.ball_vel = np.array([np.random.uniform(-5, 5), np.random.uniform(5, 10)])  # Start with upward velocity
        self.opponent_pos = np.array([3*GAME_WIDTH/4, PLAYER_RADIUS])
        self.opponent_vel = np.array([0.0, 0.0])
        
        self.score = 0
        self.opponent_score = 0
        self.done = False
        
        return self._get_obs()
        
    def step(self, action):
        # Process action
        dx = action[0] * PLAYER_SPEED_SCALE
        dy = action[2] * PLAYER_SPEED_SCALE if action[2] > 0 else 0
        
        # Update player velocity and position
        self.player_vel[0] = self.player_vel[0] * PLAYER_DAMPING + dx
        self.player_vel[1] = self.player_vel[1] * PLAYER_DAMPING + dy + GRAVITY * TIMESTEP
        
        self.player_pos += self.player_vel * TIMESTEP
        
        # Simple opponent AI
        ball_dx = self.ball_pos[0] - self.opponent_pos[0]
        self.opponent_vel[0] = np.clip(ball_dx * 0.3, -PLAYER_SPEED_SCALE, PLAYER_SPEED_SCALE)
        if self.ball_pos[1] > GAME_HEIGHT/2 and self.ball_pos[0] > GAME_WIDTH/2:
            self.opponent_vel[1] = PLAYER_SPEED_SCALE
        else:
            self.opponent_vel[1] = self.opponent_vel[1] * PLAYER_DAMPING + GRAVITY * TIMESTEP
            
        self.opponent_pos += self.opponent_vel * TIMESTEP
        
        # Update ball
        self.ball_vel[1] += GRAVITY * TIMESTEP
        self.ball_pos += self.ball_vel * TIMESTEP
        
        # Collisions
        reward = 0
        
        # Ball-ground collision
        if self.ball_pos[1] <= BALL_RADIUS:
            if self.ball_pos[0] < GAME_WIDTH/2:
                self.opponent_score += 1
                reward = -1
                self.done = True
            else:
                self.score += 1
                reward = 1
                self.done = True
            self.ball_pos[1] = BALL_RADIUS
            self.ball_vel[1] = -self.ball_vel[1] * BALL_ELASTICITY
            
        # Ball-ceiling collision
        if self.ball_pos[1] >= GAME_HEIGHT - BALL_RADIUS:
            self.ball_pos[1] = GAME_HEIGHT - BALL_RADIUS
            self.ball_vel[1] = -self.ball_vel[1] * BALL_ELASTICITY
            
        # Ball-wall collisions
        if self.ball_pos[0] <= BALL_RADIUS:
            self.ball_pos[0] = BALL_RADIUS
            self.ball_vel[0] = -self.ball_vel[0] * BALL_ELASTICITY
        elif self.ball_pos[0] >= GAME_WIDTH - BALL_RADIUS:
            self.ball_pos[0] = GAME_WIDTH - BALL_RADIUS
            self.ball_vel[0] = -self.ball_vel[0] * BALL_ELASTICITY
            
        # Ball-net collision
        if (abs(self.ball_pos[0] - GAME_WIDTH/2) < WALL_WIDTH/2 and 
            self.ball_pos[1] < WALL_HEIGHT):
            # Ball hits the net
            if self.ball_pos[0] < GAME_WIDTH/2:
                self.ball_pos[0] = GAME_WIDTH/2 - WALL_WIDTH/2 - BALL_RADIUS
            else:
                self.ball_pos[0] = GAME_WIDTH/2 + WALL_WIDTH/2 + BALL_RADIUS
            self.ball_vel[0] = -self.ball_vel[0] * BALL_ELASTICITY
            
        # Ball-player collisions
        def check_ball_player_collision(player_pos):
            dx = self.ball_pos[0] - player_pos[0]
            dy = self.ball_pos[1] - player_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < BALL_RADIUS + PLAYER_RADIUS:
                # Normal vector from player to ball
                nx = dx/dist
                ny = dy/dist
                # Relative velocity
                dvx = self.ball_vel[0] - player_pos[0]
                dvy = self.ball_vel[1] - player_pos[1]
                # Impact speed along normal
                imp = dvx*nx + dvy*ny
                # Only bounce if moving toward each other
                if imp < 0:
                    # Add some of the player's velocity to the ball
                    self.ball_vel[0] = -imp*nx * BALL_ELASTICITY + 0.3 * self.player_vel[0]
                    self.ball_vel[1] = -imp*ny * BALL_ELASTICITY + 0.3 * self.player_vel[1]
                return True
            return False
            
        check_ball_player_collision(self.player_pos)
        check_ball_player_collision(self.opponent_pos)
        
        # Constrain players
        self.player_pos[0] = np.clip(self.player_pos[0], PLAYER_RADIUS, GAME_WIDTH/2 - PLAYER_RADIUS - WALL_WIDTH/2)
        self.player_pos[1] = np.clip(self.player_pos[1], PLAYER_RADIUS, GAME_HEIGHT - PLAYER_RADIUS)
        self.opponent_pos[0] = np.clip(self.opponent_pos[0], GAME_WIDTH/2 + PLAYER_RADIUS + WALL_WIDTH/2, GAME_WIDTH - PLAYER_RADIUS)
        self.opponent_pos[1] = np.clip(self.opponent_pos[1], PLAYER_RADIUS, GAME_HEIGHT - PLAYER_RADIUS)
        
        # Constrain ball velocity
        ball_speed = np.sqrt(np.sum(self.ball_vel**2))
        if ball_speed > MAX_BALL_SPEED:
            self.ball_vel *= MAX_BALL_SPEED/ball_speed
            
        return self._get_obs(), reward, self.done, {}
        
    def _get_obs(self):
        return np.concatenate([
            self.player_pos, self.player_vel,
            self.ball_pos, self.ball_vel,
            self.opponent_pos, self.opponent_vel
        ])
        
    def render(self, mode='human'):
        if self.viewer is None:
            if mode == 'human':
                pygame.init()
                self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
                pygame.display.set_caption('Volleyball')
                self.clock = pygame.time.Clock()
            else:
                self.screen = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
                
        # Clear screen
        if mode == 'human':
            self.screen.fill((255, 255, 255))
        else:
            self.screen.fill(255)
            
        # Draw net
        if mode == 'human':
            pygame.draw.rect(self.screen, (100, 100, 100),
                           pygame.Rect(WINDOW_WIDTH/2 - WALL_WIDTH/2, 
                                     WINDOW_HEIGHT - WALL_HEIGHT,
                                     WALL_WIDTH, WALL_HEIGHT))
        else:
            cv2.rectangle(self.screen,
                         (int(WINDOW_WIDTH/2 - WALL_WIDTH/2), int(WINDOW_HEIGHT - WALL_HEIGHT)),
                         (int(WINDOW_WIDTH/2 + WALL_WIDTH/2), WINDOW_HEIGHT),
                         (100, 100, 100),
                         -1)
            
        # Draw players
        if mode == 'human':
            pygame.draw.circle(self.screen, (255, 100, 100),
                             (int(self.player_pos[0]), WINDOW_HEIGHT - int(self.player_pos[1])),
                             int(PLAYER_RADIUS))
            pygame.draw.circle(self.screen, (100, 100, 255),
                             (int(self.opponent_pos[0]), WINDOW_HEIGHT - int(self.opponent_pos[1])),
                             int(PLAYER_RADIUS))
        else:
            cv2.circle(self.screen,
                      (int(self.player_pos[0]), WINDOW_HEIGHT - int(self.player_pos[1])),
                      int(PLAYER_RADIUS),
                      (100, 100, 255),
                      -1)
            cv2.circle(self.screen,
                      (int(self.opponent_pos[0]), WINDOW_HEIGHT - int(self.opponent_pos[1])),
                      int(PLAYER_RADIUS),
                      (255, 100, 100),
                      -1)
            
        # Draw ball
        if mode == 'human':
            pygame.draw.circle(self.screen, (100, 255, 100),
                             (int(self.ball_pos[0]), WINDOW_HEIGHT - int(self.ball_pos[1])),
                             int(BALL_RADIUS))
            
            # Draw scores
            font = pygame.font.Font(None, 36)
            text = font.render(f"{self.score} - {self.opponent_score}", True, (0, 0, 0))
            text_rect = text.get_rect(center=(WINDOW_WIDTH/2, 30))
            self.screen.blit(text, text_rect)
            
            pygame.display.flip()
            self.clock.tick(60)
        else:
            cv2.circle(self.screen,
                      (int(self.ball_pos[0]), WINDOW_HEIGHT - int(self.ball_pos[1])),
                      int(BALL_RADIUS),
                      (100, 255, 100),
                      -1)
            return np.copy(self.screen)
            
    def close(self):
        if self.viewer is not None:
            pygame.quit()
            self.viewer = None
