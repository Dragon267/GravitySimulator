import matplotlib.pyplot as plt
import numpy as np
from operations import operations
import math

with open('/Users/danila/Documents/space_flight_simulator/all_angle/observation_all_angle.txt', 'r') as file:
    datax = file.readlines()

    mas = []

    for i in range(len(datax)):
        qwe = datax[i].split()
        for g in range(len(qwe)):
            qwe[g] = float(qwe[g])
        
        mas.append(qwe)
    
    datax = np.array(mas)

    file.close()

with open('/Users/danila/Documents/space_flight_simulator/all_angle/action_all_angle.txt', 'r') as file:
    datay = file.readlines()

    mas = []

    for i in range(len(datay)):
        mas.append(int(datay[i]))
    
    datay = np.array(mas)

    file.close()

print(f"datax => {datax}")
print(f"datay => {datay}")

positionx = []
positiony = []

for i in range(len(datax)):
    positionx.append(datax[i][0])
    positiony.append(datax[i][1])

plt.plot(positionx, positiony, 'ro')
plt.show()