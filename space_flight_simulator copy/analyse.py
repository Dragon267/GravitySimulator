import matplotlib.pyplot as plt
import numpy
import math
from operations import operations

with open('planet_orbit.txt', 'r') as f:
    text = f.readlines()
    f.close()

xpos = []
ypos = []

for i in range(len(text)):
    line = text[i].split(' ')

    xpos.append(float(line[0]))
    ypos.append(float(line[1]))

def Index(mas):
    index = []
    for i in range(len(mas)):
        index.append(i)
    return index


plt.plot(xpos, ypos, 'ro')

#plt.xlim(-3, 3)
#plt.ylim(-3, 3)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

plt.show()