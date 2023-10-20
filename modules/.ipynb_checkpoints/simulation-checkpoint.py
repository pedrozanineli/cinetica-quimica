import os
import lmfit
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

from modules.particle import Particle

class Simulation:

    def __init__(self, n, radius=1, styles=None, prob = 0.5):
        self.init_particles(n, radius, styles)
        self.a_s, self.b_s, self.time = [], [], []
        self.prob_reac = 1-prob

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
                
                particle = Particle(x, y, vx, vy, 'darkorange', rad, styles)
                
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
            
            merged_radius = np.sqrt(p1.radius**2 + p2.radius**2)
            merged_x = (p1.x + p2.x) / 2
            merged_y = (p1.y + p2.y) / 2
            merged_vx = (p1.vx + p2.vx) / 2
            merged_vy = (p1.vy + p2.vy) / 2
            
            merged_particle = Particle(merged_x, merged_y, merged_vx, merged_vy, 'royalblue', merged_radius)

            p1.merged = True
            p2.merged = True

            return merged_particle
    
        particles_to_remove = []
        particles_to_add = []

        pairs = combinations(range(len(self.particles)), 2)
        for i, j in pairs:
            
                if not (self.particles[i].merged or self.particles[j].merged) and np.random.random() > self.prob_reac:
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
        
        plt.rcParams['font.family'] = 'serif'
        
        fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,5))
        
        a_count,b_count=0,0
        
        ax1.set_title(r'$2A \rightarrow A_2$')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        self.vs = []
        
        for particle in self.particles:
            circle = Circle(xy=particle.r, radius=particle.radius, edgecolor=particle.color, fill=False)
            ax1.add_patch(circle)
            self.vs.append(np.sqrt(particle.vx**2+particle.vy**2))
            
            if particle.color=='royalblue':
                a_count += 1
            else:
                b_count += 1
                
        ax2.hist(self.vs)
        ax2.set_xlim(0,0.25)
        ax2.set_ylabel('Frequência',fontsize=16)
        ax2.set_xlabel('Velocidade',fontsize=16)

        self.a_s.append(a_count)
        self.b_s.append(b_count)
        self.time.append(i)
        
        ax3.scatter(self.time,self.a_s,label='$A$',color='royalblue')
        ax3.scatter(self.time,self.b_s,label='$A_2$',color='darkorange')
        ax3.set_xlim(0,frames)
        ax3.set_xlabel('Tempo',fontsize=16)
        ax3.set_ylabel('Concentração',fontsize=16)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig('figs/{:03d}.png'.format(i))
        plt.close()
    
    def init(self):

        self.circles = []
        for particle in self.particles:
            self.circles.append(particle)
        return self.circles
    
    def run(self,frames,display_gif=True,display_fit=True,save_gif=True):
        
        self.init()
        progresso = tqdm(total=frames)
        
        for i in range(frames):            
            self.draw(i,frames)
            self.advance_animation(0.1)
            progresso.update(1)
        
        progresso.close()

        if save_gif:
            image_paths = ['figs/{:03d}.png'.format(i) for i in range(frames)]
            images = []

            print('Gerando o gif')
            for path in image_paths:
                image = Image.open(path)
                images.append(image)

            output_path = 'sim.gif'
            imageio.mimsave(output_path, images, duration=0.01)

            if display_gif:
                display(ipyimage(filename=output_path))
        
        def fit(x, k, C, D):
            return C*np.exp(-k*x) + D
        
        funcao_fit_model=lmfit.Model(fit)
        funcao_fit_model.set_param_hint('D',value=30, vary=True)
        funcao_fit_model.set_param_hint('C',value=150, vary=True)
        funcao_fit_model.set_param_hint('k',value=.05, vary=True)

        param = funcao_fit_model.make_params()
        results = funcao_fit_model.fit(self.b_s, x = list(range(frames)), params = param)
        k = results.params['k'].value
        D = results.params['D'].value
        C = results.params['C'].value
        x_fit = np.linspace(0, frames)
        y_fit_A = fit(x_fit, k, C, D)
        y_fit_B = (-np.array(y_fit_A) + self.b_s[0])/2
        
        if display_fit:
            plt.figure(figsize=(10,4))

            plt.subplot(121)
            plt.scatter(self.time,self.a_s,label='$A$',color='royalblue')
            plt.scatter(self.time,self.b_s,label='$A_2$',color='darkorange')
            plt.xlim(0,frames)

            plt.plot(x_fit, y_fit_A, linewidth = 2.5, linestyle = '--', c='0')
            plt.plot(x_fit, y_fit_B, linewidth = 2.5, linestyle = ':', c='0')

            plt.xlabel('Tempo',fontsize=16)
            plt.ylabel('Concentração',fontsize=16)
            plt.legend()

            derivada_A = [y_fit_A[i] - y_fit_A[i - 1] for i in range(1,len(y_fit_A))]
            derivada_B = [y_fit_B[i] - y_fit_B[i - 1] for i in range(1,len(y_fit_B))]

            plt.subplot(122)
            plt.plot(x_fit[:-1],derivada_A,label='[$A$]')
            plt.plot(x_fit[:-1],derivada_B,label='[$A_2$]')
            plt.xlabel('Tempo',fontsize=16),plt.ylabel('d[]/dt',fontsize=16)
            plt.legend()

            plt.tight_layout()
        
        return k