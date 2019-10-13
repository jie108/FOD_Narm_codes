import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve', color = 'rb')
ax.legend()

#plt.show(block=True)
plt.show()


import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

xy = (np.random.random((1000, 3)) - 0.5).cumsum(axis=0)

# Reshape things so that we have a sequence of:
# [[(x0,y0),(x1,y1)],[(x0,y0),(x1,y1)],...]
xy = xy.reshape(-1, 1, 3)
segments = np.hstack([xy[:-1], xy[1:]])

fig, ax = plt.subplots()
coll = LineCollection(segments, cmap=plt.cm.gist_ncar)
coll.set_array(np.random.random(xy.shape[0]))

ax.add_collection(coll)
ax.autoscale_view()

plt.show()


import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm    
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Generate random data
np.random.seed(1)
n = 5 # number of data points
#set x,y,z data
x = np.random.uniform(0, 1, n)
y  = np.random.uniform(0, 1, n)
z = np.arange(0,n)

# Create a colormap for red, green and blue and a norm to color
# f' < -0.5 red, f' > 0.5 blue, and the rest green
#cmap = ListedColormap(['r', 'g', 'b'])
#norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)

#################
### 3D Figure ###
#################

# Create a set of line segments
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create the 3D-line collection object
#lc = Line3DCollection(segments, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0, n))
#lc = Line3DCollection(segments, colors=["#2744F3FF","#3154ECFF","#4553E7FF"])
lc = Line3DCollection(segments, colors=list([(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)]))
lc.set_array(z) 
lc.set_linewidth(2)

#plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_zlim(0, max(z))
plt.title('3D-Figure')
#ax.add_collection3d(lc, zs=z, zdir='z')
ax.add_collection3d(lc)

plt.show()


import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt   
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

fig = plt.figure();
ax = fig.gca(projection = '3d')
X = [(0,0,0,1,0),(0,0,1,0,0),(0,1,0,0,0)]
points = np.array([X[0], X[1], X[2]]).T.reshape(-1, 1, 3)
r = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.5, 0.5, 0.5, 1.0)];

segs = np.concatenate([points[:-1], points[1:]], axis = 1)
ax.add_collection(Line3DCollection(segs,colors=list(r)))
ax.add_collection(Line3DCollection(segments,colors=list(r)))

plt.show()
