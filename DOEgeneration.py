from scipy.stats import qmc
import numpy as np
import matplotlib.pylab as plt

sampler = qmc.LatinHypercube(d=4,optimization="random-cd")
sample = sampler.random(n=25)
l_bounds = [10.0,13.0,10.0,12.0]
u_bounds = [20.0,20.0,17.0,20.0]
# l_bounds = [10.0,9.0]
# u_bounds = [22.0,18.0]
sample_scaled = qmc.scale(sample,l_bounds,u_bounds)
print(sample_scaled)
np.savetxt('4DOF_DOEarrayinsidelarge_2_18', sample_scaled, delimiter = ' , ')
 
# plt.figure()
# plt.plot(sample_scaled[:,0],sample_scaled[:,1],'.')
# # plt.axis('square')
# plt.show()