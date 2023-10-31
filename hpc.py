import numpy as np
import matplotlib.pyplot as plt
from modules.simulation import Simulation
from sklearn.linear_model import LinearRegression

nparticles = 150
radii = np.ones(nparticles)*0.01
p = 0.6

styles = {'linewidth': .1, 'fill': None}
sim = Simulation(nparticles, radii, styles, prob = p)

sim.run(frames=300,save_gif=False,catalisador=False)
