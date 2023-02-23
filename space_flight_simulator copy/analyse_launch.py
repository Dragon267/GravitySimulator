import matplotlib.pyplot as plt
import numpy as np
from operations import operations
import math

with open('data_of_launch.txt', 'r') as file:
    datax = file.readlines()

    mas = []

    for i in range(len(datax)):
        qwe = datax[i].split()
        for g in range(len(qwe)):
            qwe[g] = float(qwe[g])
        
        mas.append(qwe)
    
    datax = np.array(mas)

    file.close()

with open('action_data_launch.txt', 'r') as file:
    datay = file.readlines()

    mas = []

    for i in range(len(datay)):
        mas.append(int(datay[i]))
    
    datay = np.array(mas)

    file.close()

print(f"datax => {datax}")
print(f"datay => {datay}")

def rotate(value1, value2):
    positionx = []
    positiony = []
    position = []

    for i in range(len(datax)):
        positionx.append(datax[i][value1])
        positiony.append(datax[i][value2])

        position.append([datax[i][value1], datax[i][value2]])

    oper = operations()

    new_position_x = []
    new_position_y = []

    for i in range(len(positionx)):
        if positionx[i] != 0 and positiony[i] != 0:
            position = [positionx[i], positiony[i]]
            unit_vec = oper.divide(position, oper.magnitude(position))

            x = unit_vec[0] * np.cos(angle) - unit_vec[1] * np.sin(angle)
            y = unit_vec[0] * np.sin(angle) + unit_vec[1] * np.cos(angle)

            new_unit_vec = [x, y]

            new_position = oper.multi(new_unit_vec, oper.magnitude(position))

            new_position_x.append(new_position[0])
            new_position_y.append(new_position[1])
        else:
            new_position_x.append(0)
            new_position_y.append(0)

    return new_position_x, new_position_y


angle = 45
step_angle = 45
value_angle = 90

for i in range(7):
    px, py = rotate(0, 1)
    vx, vy = rotate(2, 3)

    new_angle = (angle + value_angle) % 360
    angle += (step_angle * math.pi / 180)
    angle = angle % 360

    with open('/Users/danila/Documents/space_flight_simulator/all_angle/action_all_angle.txt', 'a') as file:
        for g in range(len(datay)):
            file.write(str(datay[g]) + '\n')
        
        file.close()
    
    with open('/Users/danila/Documents/space_flight_simulator/all_angle/observation_all_angle.txt', 'a') as file:
        for g in range(len(px)):
            file.write(str(px[g]) + " " + str(py[g]) + " " + str(vx[g]) + " " + str(vy[g]) + " " + str(new_angle) + " " + str(datax[g][5]) + '\n')

        file.close()
