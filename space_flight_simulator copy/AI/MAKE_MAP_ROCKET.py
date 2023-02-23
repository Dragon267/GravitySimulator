import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import numpy as np

'''

input_layer = [bodies[rocket].px, bodies[rocket].py, bodies[rocket].vx, bodies[rocket].vy, self.angle, self.START]

            if self.DRIVE == True:
                qwe = model.predict([input_layer])
                pred = np.argmax(qwe[0])

                if pred == 1:
                    self.START = 1
                elif pred == 2:
                    self.START = 0
                elif pred == 3:
                    self.angle += 5
                elif pred == 4:
                    self.angle -= 5

'''

vx = 5
vy = 5
angle = 90
START = 1

model = tf.keras.models.load_model('/Users/danila/Documents/space_flight_simulator/rocket_alpha_1.model')

def Index(l, r, step):
    ans = []
    for i in range(l, r,step):
        ans.append(i)
    return ans

def function(x, y):
    input_layer = [x, y, vx, vy, angle, START]
    qwe = model.predict([input_layer])
    pred = np.argmax(qwe[0])

    return pred

def ModelFunction(x, y):
    ans = []
    for i in range(len(x)):
        for g in range(len(y)):
            xx = x[i]
            yy = y[i]

            action = function(xx, yy)


x = Index(-10, 10, 1)
y = Index(-10, 10, 1)
z = []

'''

if pred == 1:
    self.START = 1
elif pred == 2:
    self.START = 0
elif pred == 3:
    self.angle += 5
elif pred == 4:
    self.angle -= 5

'''

for i in range(len(x)):
    for g in range(len(y)):
        res = function(x[i], y[g])
        if (res == 1):
            plt.plot(x[i], y[g], 'b.')
        elif (res == 2):
            plt.plot(x[i], y[g], 'r.')
        elif (res == 3):
            plt.plot(x[i], y[g], 'g.')
        elif (res == 4):
            plt.plot(x[i], y[g], 'm.')

#fig = plt.figure()

#ax = fig.add_subplot(111, projection='3d')

#plt.plot(x, y, 'b.')

plt.show()