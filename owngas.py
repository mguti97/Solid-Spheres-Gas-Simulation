import numpy as np
import math
import random as rd
from random import random
import matplotlib.pyplot as plt
from numpy import linalg as LA
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from scipy import stats


N = 500
Matom = 6.64e-24 #oxigen mass
# Matom = 7.68*1.66054e-27 
Ratom = 0.005 # wildly exaggerated size of helium atom
k = 1.38064e-23 # Boltzmann constant
T = 300.
L = 1.
T_MAX = 3
dt = 1e-3
A = 6*L**2


#####################################inicialitzacio
def init_part(Ntot, rad, mass):
	r = np.zeros((Ntot,3))
	v = np.zeros((Ntot,3))

	vrms = np.sqrt(3*k*T/mass)

	for i in range(N):
		r[i][0] = (L/2)*rd.uniform(-1,1)
		r[i][1] = (L/2)*rd.uniform(-1,1)
		r[i][2] = (L/2)*rd.uniform(-1,1)

		theta = np.pi*random()
		phi = 2*np.pi*random()

		v[i][0] = vrms*np.sin(theta)*np.cos(phi)
		v[i][1] = vrms*np.sin(theta)*np.sin(phi)
		v[i][2] = vrms*np.cos(theta)


	return r, v

pos, vel = init_part(N, Ratom, Matom)
CM = (0,0,0)
visualpart = (0,0,0)

#####################################temperatura experimental
def Tempexp(PARTICLES, v, mass):
	suma = 0
	for i in range(PARTICLES):
		suma = suma + (LA.norm(v[i]))**2

	temperature = suma*mass/(3*PARTICLES*k)

	return temperature

#####################################walls
def walls(r, v, i, x):
	pex = 0
	for k in range(3):
		if abs(r[i][k]) + Ratom > x/2:
			v[i][k] *= -1
			if r[i][k] < 0:
				r[i][k] = -x/2 + Ratom
			else :
				r[i][k] = x/2 - Ratom

			pex += 2 * Matom * abs(v[i][k])
	return pex

#####################################colision
def mod(v):
    """
        computes the squared sum over the last axis of the numpy.ndarray v
    """
    return np.sum(v * v, axis=-1)

def check_collision(r, v, i):  #Check for collisions with other particles of te same gas
	dists = np.sqrt(mod(r - r[:,np.newaxis])) 
	cols2 = (0 < dists) & (dists < 2*Ratom)
	idx_i, idx_j = np.nonzero(cols2)
	# ***possibility to simplify this *** #
	for i, j in zip(idx_i, idx_j):
		if j < i:
		    # skip duplications and same particle
		    continue 

		rij = r[i] - r[j]
		d = mod(rij)
		vij = v[i] - v[j]
		dv = np.dot(vij, rij) * rij / d
		v[i] -= dv
		v[j] += dv
			
		# update the positions so they are no longer in contact
		r[i] += dt * v[i]
		r[j] += dt * v[j]

#####################################funcio iterable
pexchange = []
TEXP = []
hist = []
def update(j, r, v, lines):
	pex = 0
	modv = 0
	for i in range(N):
		pex += walls(pos, vel, i, L)
		r[i] += v[i]*dt
		modv = modv + LA.norm(v[i])
	check_collision(pos, vel, i)

	TEXP.append(Tempexp(N, v, Matom))
	pexchange.append(pex)
	average_v = modv/N
	hist.append(average_v)
	
	CM = np.sum(r, axis=0) / N
	visualpart = r[1]
		
	lines[0].set_data(r[:,0], r[:,1])
	lines[0].set_3d_properties(r[:,2])

	lines[1].set_data([CM[0]], [CM[1]])
	lines[1].set_3d_properties([CM[2]])

	lines[2].set_data([visualpart[0]], [visualpart[1]])
	lines[2].set_3d_properties([visualpart[2]])
	

	return lines

##########################################animacio
fig = plt.figure()
ax = p3.Axes3D(fig)

ax.set_xlim3d([-L/2, L/2])
ax.set_xlabel('X')

ax.set_ylim3d([-L/2, L/2])
ax.set_ylabel('Y')

ax.set_zlim3d([-L/2, L/2])
ax.set_zlabel('Z')

lines = []
lines.append(ax.plot(pos[:,0], pos[:,1], pos[:,2], ls='None', marker='.', label = 'Particles at $T_0=%.2f\,K$'%T)[0])
lines.append(ax.plot([CM[0]], [CM[1]], [CM[2]], marker='o', color='r', label='Center of masses')[0])
lines.append(ax.plot([visualpart[0]], [visualpart[1]], [visualpart[2]], marker='o', color = 'orange')[0])

ani = FuncAnimation(fig, update, fargs=(pos, vel, lines), frames=np.linspace(0, T_MAX-dt, int(T_MAX / dt)), blit=True, interval=0.0005, repeat=False)

plt.legend(loc = 'upper left')
plt.show()


##########################################presio
theorical = []
pressure = []
for i in range(len(pexchange)):
	theorical.append(N*k*T/(L**3))
	pressure.append(pexchange[i]/(A*dt))

pmitjana = []
for i in range(len(pressure)):
	pmitjana.append(sum(pressure)/len(pressure))

t = np.linspace(0, T_MAX, len(pressure))
plt.plot(t, pressure, label ='presió experimental')
plt.plot(t, theorical, label = 'Presió teòrica')
plt.plot(t, pmitjana, label = 'Presió mitjana experimental')
plt.legend()
plt.show()

############################################histograma
modvel = []
for i in range(len(vel)):
	modvel.append(np.sqrt(mod(vel[i])))
def fmb(m, T, v):
	return (m/(2.*np.pi*k*T))**1.5 * 4.*np.pi*v*v * np.exp((-m*v*v)/(2.*k*T))  #Maxwell-Boltzman distribution

v = np.linspace(0.0, max(modvel), 1000)

plt.plot(v, fmb(Matom, T, v), 'r-')
n, bins, patches = plt.hist(modvel, 40, facecolor='g', alpha=0.5, density = 1)
plt.xlabel('Speed -- v (m/s)')
plt.ylabel('Probability -- f(v)')
plt.title('Histogram of Maxwell-Boltzmann speed distribution')
plt.grid(True)
plt.show()


##########################################velocity evolution
vmed = []
vrms = []

for i in range(len(t)):
	vmed.append(np.sqrt(8 * k * T / (np.pi * Matom)))
	vrms.append(np.sqrt(3*k*T/Matom))

plt.plot(t, hist, label = 'Experimental')
plt.plot(t, vmed, label = 'vmed')
plt.plot(t, vrms, label = 'vrms')
plt.xlabel('Speed -- v (m/s)')
plt.ylabel('time -- t (s)')
plt.legend()
plt.show()

##########################################print data
print(pmitjana[0])
print((pmitjana[0]-theorical[0])/max(theorical[0], pmitjana[0]))
