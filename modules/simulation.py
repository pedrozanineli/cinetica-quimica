import os
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from itertools import combinations
from scipy.optimize import curve_fit
from matplotlib.patches import Circle
from IPython.display import display
from IPython.display import Image as ipyimage
from statsmodels.nonparametric.smoothers_lowess import lowess

from modules.particle import Particle

class Simulation:

    def __init__(self, n, radius=0.01, styles=None):
        self.init_particles(n, radius, styles)
        self.a_s, self.b_s, self.time, self.vs = [], [], [], []

    def init_particles(self, n, radius, styles=None):

        try:
            iterator = iter(radius)
            assert n == len(radius)
        except TypeError:
            def r_gen(n, radius):
                for i in range(n):
                    yield radius
            radius = r_gen(n, radius)

        self.n = n
        self.particles = []

        
        for i, rad in enumerate(radius):
            while True:
                x, y = rad + (1 - 2*rad) * np.random.random(2)
                vr = 0.1 * np.random.random() + 0.05
                vphi = 2*np.pi * np.random.random()
                vx, vy = vr * np.cos(vphi), vr * np.sin(vphi)
                
                particle = Particle(x, y, vx, vy, 'r', rad, styles)
                
                for p2 in self.particles:
                    if p2.overlaps(particle):
                        break
                else:
                    self.particles.append(particle)
                    break
    
    def handle_collisions(self):

        def change_velocities(p1, p2):

            m1, m2 = p1.radius**2, p2.radius**2
            M = m1 + m2
            r1, r2 = p1.r, p2.r
            d = np.linalg.norm(r1 - r2)**2
            v1, v2 = p1.v, p2.v
            u1 = v1 - 2*m2 / M * np.dot(v1-v2, r1-r2) / d * (r1 - r2)
            u2 = v2 - 2*m1 / M * np.dot(v2-v1, r2-r1) / d * (r2 - r1)
            p1.v = u1
            p2.v = u2
        
        def merge_particles(p1, p2):
            
            # merged_radius = np.sqrt(p1.radius**2 + p2.radius**2)
            merged_x = (p1.x + p2.x) / 2
            merged_y = (p1.y + p2.y) / 2
            merged_vx = (p1.vx + p2.vx) / 2
            merged_vy = (p1.vy + p2.vy) / 2
            
            merged_particle = Particle(merged_x, merged_y, merged_vx, merged_vy, 'royalblue', p1.radius)

            p1.merged = True
            p2.merged = True

            return merged_particle
    
        particles_to_remove = []
        particles_to_add = []

        pairs = combinations(range(len(self.particles)), 2)
        for i, j in pairs:
            
            if not (self.particles[i].merged or self.particles[j].merged):
                if self.particles[i].overlaps(self.particles[j]):
                    merged_particle = merge_particles(self.particles[i], self.particles[j])
                    merged_particle.merged = True
                    particles_to_add.append(merged_particle)
                    particles_to_remove.extend([i, j])
            else:
                if self.particles[i].overlaps(self.particles[j]):
                    change_velocities(self.particles[i], self.particles[j])

        self.particles = [p for idx, p in enumerate(self.particles) if idx not in particles_to_remove]
        self.particles.extend(particles_to_add)

    def advance_animation(self, dt):

        for i, p in enumerate(self.particles):
            p.advance(dt)
            self.circles[i].center = p.r
        self.handle_collisions()
        return self.circles
    
    def draw(self,i,frames):
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(20,5))
        
        a_count,b_count=0,0
        
        ax1.set_title(r'$2A \rightarrow A_2$')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        for particle in self.particles:
            circle = Circle(xy=particle.r, radius=particle.radius, edgecolor=particle.color, fill=False)
            ax1.add_patch(circle)
            # vs.append(np.sqrt(particle.vx**2+particle.vy**2))
            self.vs.append(np.sqrt(particle.vx**2+particle.vy**2))
            
            if particle.color=='royalblue':
                a_count += 1
            else:
                b_count += 1
                
        ax2.hist(self.vs)
        ax2.set_xlim(0,0.25)
        ax2.set_ylabel('Frequência')
        ax2.set_xlabel('Velocidade')

        self.a_s.append(a_count)
        self.b_s.append(b_count)
        self.time.append(i)
        
        ax3.scatter(self.time,self.a_s,label='A')
        ax3.scatter(self.time,self.b_s,label='B')
        ax3.set_xlim(0,frames)
        ax3.set_xlabel('Tempo')
        ax3.set_ylabel('[]')
        ax3.legend()
        
        if len(self.a_s)>1 and len(self.b_s)>1:            
            smoothed_a = lowess(self.a_s, self.time, frac=0.1)
            smoothed_a = smoothed_a[:, 1]
            derivative_a = np.gradient(smoothed_a, self.time)
            
            smoothed_b = lowess(self.b_s, self.time, frac=0.1)
            smoothed_b = smoothed_b[:, 1]
            derivative_b = np.gradient(smoothed_b, self.time)
            
            ax4.plot(self.time,derivative_a,label='A')
            ax4.plot(self.time,derivative_b,label='B')
            ax4.set_xlim(0,frames)
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('figs/{:03d}.png'.format(i))
        plt.close()
    
    def init(self):

        self.circles = []
        for particle in self.particles:
            self.circles.append(particle)
        return self.circles
    
    def run(self,frames):
        
        self.init()
        progresso = tqdm(total=frames)
        
        for i in range(frames):
            self.draw(i,frames)
            self.advance_animation(0.1)
            progresso.update(1)
        
        progresso.close()

        image_paths = ['figs/{:03d}.png'.format(i) for i in range(frames)]

        images = []
        
        print('Gerando o gif')
        for path in image_paths:
            image = Image.open(path)
            images.append(image)

        output_path = 'sim.gif'
        imageio.mimsave(output_path, images, duration=0.01)
        
        display(ipyimage(filename=output_path))