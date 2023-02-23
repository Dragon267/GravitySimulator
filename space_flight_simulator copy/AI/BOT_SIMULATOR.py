import pygame
import sys
from operations import operations
import math
import numpy as np

oper = operations()

# The gravitational constant G
G = 6.67428e-11

# Assumed scale: 100 pixels = 1AU.
AU = (149.6e6 * 1000)     # 149.6 million km, in meters.
#SCALE = 250 / AU

LINE_SCALE = 10

RADIUS_SCALE = 500

class BOT_SIMULATOR:

    START = 0

    #SCALE = 35000 / AU
    SCALE = 35000 / AU

    MOVEMENT = 10
    ZOOMING = 10000

    timestep = 24 * 3600

    DRIVE = True

    x, y = 0, 0

    angle = 0

    def update_info(self, step, bodies):
        """(int, [Body])
        
        Displays information about the status of the simulation.
        """
        print('Step #{}'.format(step))
        for body in bodies:
            s = '{:<8}  Pos.={:>6.2f} {:>6.2f} Vel.={:>10.3f} {:>10.3f}'.format(
                body.name, body.px/AU, body.py/AU, body.vx, body.vy)
            print(s)
        print()

    def check_existance(self, mas, name):
        ans = False
        for i in range(len(mas)):
            if mas[i] == name:
                return True
        return False


    def loop(self, bodies, model, main=None, plot=None, rocket=None, plot_actions=None):
        """([Body])

        Never returns; loops through the simulation, updating the
        positions of all the provided bodies.
        """

        pygame.init()

        clock = pygame.time.Clock()

        screen = pygame.display.set_mode((600, 600))

        pygame.mouse.set_visible(0)

        pygame.display.set_caption('Space Age Game')

        background = (24, 39, 95)

        step = 1

        answer = []

        while True:
            for body in bodies:
                body.prx = body.px
                body.pry = body.py
                body.prvx = body.vx
                body.prvy = body.vy


            RADIUS_SCALE = 1 / self.SCALE

            print(f"x => {self.x}, y = {self.y}")
            print(f"SCALE => {self.SCALE}")

            #print(f'direction x=>{bodies[rocket].direction[0]}, y=>{bodies[rocket].direction[1]}')
            print(f'timestep => {self.timestep}')
            print(f"dist => {oper.magnitude(oper.minus([body.px, body.py], [bodies[main].px, bodies[main].py]))}")
            print(f"velocity of rocket vx => {body.vx}, vy => {body.vy}")
            print(f'mass of rocket => {bodies[rocket].mass}')

            clock.tick(60)

            screen.fill(background)

            self.update_info(step, bodies)

            step += 1

            force = {}
            for body in bodies:
                # Add up all of the forces exerted on 'body'.
                total_fx = total_fy = 0.0
                for other in bodies:
                    # Don't calculate the body's attraction to itself
                    if body is other:
                        continue
                    fx, fy = body.attraction(other)
                    total_fx += fx
                    total_fy += fy

                # Record the total force exerted.
                force[body] = (total_fx, total_fy)

            # Update velocities based upon on the force.
            for body in bodies:

                #check = self.check_existance(constant, body.name)

                #if check == True:
                #    continue

                fx, fy = force[body]
                body.vx += fx / body.mass * self.timestep
                body.vy += fy / body.mass * self.timestep

                # Update positions
                body.px += body.vx * self.timestep
                body.py += body.vy * self.timestep
                #body.goto(body.px*SCALE, body.py*SCALE)
                #body.dot(3)

                if plot != None and plot == body.name:
                    with open('planet_orbit.txt', 'a') as f:
                        f.write(str(body.px) + " " + str(body.py) + "\n")
                        print("inserted !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    f.close()
                #if body.name == 'Moon':
                #    with open('data_moon.txt', 'a') as f:
                #        f.write(str(body.px) + " " + str(body.py) + "\n")

                #print(f"res => {body.radius * self.SCALE}")

                if main == None:
                    pygame.draw.circle(screen, body.color, (300 + body.px*self.SCALE, 300 + body.py*self.SCALE), body.radius / RADIUS_SCALE, 20)  
                if body.name == 'Rocket':
                    #print(f"previus position x => {body.prx}, y => {body.pry}")

                    if (oper.magnitude(oper.minus([body.px, body.py], [bodies[main].px, bodies[main].py]))) <= bodies[main].radius - 1:
                        position = oper.minus([body.px, body.py], [bodies[main].px, bodies[main].py])
                        mag = oper.magnitude(position)
                        unit_vec = oper.divide(position, mag)
                        result_vec = oper.plus(oper.multi(unit_vec, bodies[main].radius), [bodies[main].px, bodies[main].py])


                        #print(f"new_dist => {oper.magnitude(oper.minus([body.px, body.py], [bodies[main].px, bodies[main].py]))}")

                        body.px = result_vec[0]
                        body.py = result_vec[1]

                        body.vx = 0
                        body.vy = 0

                        #print(f"velocity QWERTY x={body.vx}, y={body.vy}")
                    
                    #pygame.draw.polygon(screen, body.color, ((body.px, body.py), (body.px - body.radius,body.py - body.radius), (body.px + body.radius,body.py - body.radius)))

                    #pygame.draw.circle(screen, body.color, (300 + (bodies[main].px - body.px) * self.SCALE + self.x, 300 + (bodies[main].py - body.py) * self.SCALE + self.y), body.radius, 20)
                    pygame.draw.line(screen, body.color, (300 + (bodies[main].px - body.px) * self.SCALE + self.x, 300 + (bodies[main].py - body.py) * self.SCALE + self.y), (300 + (bodies[main].px - body.px) * self.SCALE + self.x + body.direction[0] * LINE_SCALE, 300 + (bodies[main].py - body.py) * self.SCALE + self.y + body.direction[1] * LINE_SCALE), 5)
                    
                else:
                    if body.name == bodies[main].name:
                        pygame.draw.circle(screen, body.color, (300 + self.x, 300 + self.y), body.radius / RADIUS_SCALE, 20)  
                    else:
                        pygame.draw.circle(screen, body.color, (300 + (bodies[main].px - body.px) * self.SCALE + self.x, 300 + (bodies[main].py - body.py) * self.SCALE + self.y), body.radius / RADIUS_SCALE, 20)                

                #answer.append([body.px * SCALE, body.py * SCALE])

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


            action = -1 # no actions
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.y -= self.MOVEMENT
                    elif event.key == pygame.K_s:
                        self.y += self.MOVEMENT
                    elif event.key == pygame.K_a:
                        self.x -= self.MOVEMENT
                    elif event.key == pygame.K_d:
                        self.x += self.MOVEMENT
                    elif event.key == pygame.K_p:
                        self.MOVEMENT += 10
                    elif event.key == pygame.K_l:
                        self.MOVEMENT -= 10
                    elif event.key == pygame.K_z:
                        self.SCALE = (self.SCALE * AU + self.ZOOMING) / AU
                    elif event.key == pygame.K_x:
                        self.SCALE = (self.SCALE * AU - self.ZOOMING) / AU
                    elif event.key == pygame.K_b:
                        self.ZOOMING *= 2
                    elif event.key == pygame.K_n:
                        self.ZOOMING /= 2
                    elif event.key == pygame.K_y:
                        self.timestep *= 2
                    elif event.key == pygame.K_h:
                        self.timestep /= 2
                    elif event.key == pygame.K_i:
                        self.START = 1

                        action = 1 # launch
                    elif event.key == pygame.K_j:
                        self.START = 0

                        action = 2 # stop launch
                    elif event.key == pygame.K_9:
                        self.angle += 5

                        action = 3 # turning
                    elif event.key == pygame.K_0:
                        self.angle -= 5

                        action = 4 # turning
                    elif event.key == pygame.K_1:
                        bodies[rocket].thrust -= 100
                    elif event.key == pygame.K_2:
                        bodies[rocket].thrust += 100
                    
                    elif event.key == pygame.K_5:
                        self.DRIVE = True
                    elif event.key == pygame.K_6:
                        self.DRIVE = False
            
            bodies[rocket].direction[0] = math.cos(self.angle * math.pi / 180)
            bodies[rocket].direction[1] = math.sin(self.angle * math.pi / 180)

            if self.START == 1:
                #velocity = (bodies[rocket].fuel_velocity * bodies[rocket].fuel_mass * self.timestep) / bodies[rocket].mass
                #bodies[rocket].mass -= bodies[rocket].fuel_mass * self.timestep
                #res = oper.multi(bodies[rocket].direction, velocity)
                velocity = (bodies[rocket].thrust * self.timestep) / bodies[rocket].mass * self.timestep
                res = oper.multi(bodies[rocket].direction, velocity)

                bodies[rocket].vx += res[0]
                bodies[rocket].vy += res[1]

                print("FAST!!!!")
            
            pygame.display.update()

