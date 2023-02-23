import tensorflow as tf

class Extract:
    def extract(self):

        with open('/Users/danila/Documents/space_flight_simulator/AI/good_launch/actions.txt', 'r') as file:
            text_action = file.readlines()
            file.close()

        with open('/Users/danila/Documents/space_flight_simulator/AI/good_launch/observations.txt', 'r') as file:
            text_observation = file.readlines()
            file.close()

        #print(text_action[0])
        #rint(text_observation[0])

        # px py vx vy angle START

        datax = []

        for i in range(len(text_observation)):
            mas = text_observation[i].split()
            add = []
            for g in range(len(mas)):
                add.append(float(mas[g]))
            
            datax.append(add)

        #print(datax[0])

        datay = []

        for i in range(len(text_action)):
            value = int(text_action[i])

            qwe = [0,0,0,0,0]

            if value == -1:
                datay.append(0)

                continue

            datay.append(value)
        
        return [datax, datay]

