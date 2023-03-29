import matplotlib.pylab as plt
import numpy as np

xshift = 3.5
yshift = 2.2 #center point of testing area

scale = 1.6 #radius of turns

theta = np.radians(90.0)

r = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
quant = 400 #number of points in each half circle
delt = (np.pi*scale)/(quant-1)

top = np.empty((quant,2))

for i in range(len(top)):
    ang = (np.pi/(quant-1))*i
    top[i,0] = np.cos(ang) * scale
    top[i,1] = np.sin(ang) * scale
bottom = -top
gap = 0
gapcount = 0
while gap < (scale*1.0):
    gapcount += 1
    gap += delt
top[:,1] += gap
bottom[:,1] -= gap
left = np.empty(((2*gapcount)-1,2))
left[:,0] = -scale
for i in range(len(left)):
    left[i,1] = (gap-delt)-(i*delt)
right = -left
track = np.vstack((top,left,bottom,right))

rottrack = track
for i in range(len(track)):
    rottrack[i,:] = r.dot(track[i,:])

rottrack[:,0] = rottrack[:,0] + xshift
rottrack[:,1] = (rottrack[:,1] + yshift)

np.savetxt('trackplot_12_19_small_corrected', rottrack, delimiter = ' , ')
plt.figure()
plt.plot(rottrack[:,0],rottrack[:,1],'.')
plt.axis('square')
plt.show()