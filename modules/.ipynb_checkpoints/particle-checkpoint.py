import math
import numpy as np

class Particle:

    def __init__(self, x, y, vx, vy, color, radius=0.01, styles=None):
        
        self.r = np.array((x, y))
        self.v = np.array((vx, vy))
        self.radius = radius
        self.color = color
        self.styles = styles
        self.merged = False
        
        if not self.styles:
            self.styles = {'edgecolor': 'royalblue', 'fill': False}

    @property
    def x(self):
        return self.r[0]
    @x.setter
    def x(self, value):
        self.r[0] = value
    @property
    def y(self):
        return self.r[1]
    @y.setter
    def y(self, value):
        self.r[1] = value
    @property
    def vx(self):
        return self.v[0]
    @vx.setter
    def vx(self, value):
        self.v[0] = value
    @property
    def vy(self):
        return self.v[1]
    @vy.setter
    def vy(self, value):
        self.v[1] = value

    def overlaps(self, other):
                
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2) <= (self.radius + other.radius)
        # return np.hypot(*(self.r*2 - other.r*2)) < self.radius*2 + other.radius*2

    def advance(self, dt):

        self.r += self.v * dt

        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx
        if self.x + self.radius > 1:
            self.x = 1-self.radius
            self.vx = -self.vx
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy
        if self.y + self.radius > 1:
            self.y = 1-self.radius
            self.vy = -self.vy