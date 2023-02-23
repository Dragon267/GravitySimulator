import math
from Body import Body
import pygame
import sys
from gravity import gravity

# The gravitational constant G
G = 6.67428e-11

# Assumed scale: 100 pixels = 1AU.
AU = (149.6e6 * 1000)     # 149.6 million km, in meters.
#SCALE = 250 / AU
SCALE = 35000 / AU
#SCALE = 250000 / AU

def main():
    sun = Body()
    sun.name = 'Sun'
    sun.mass = 1.98892e30
    sun.color = (255, 255, 0)
    sun.radius = 1392700

    earth = Body()
    earth.name = 'Earth'
    earth.mass = 5.9742e24#5.9742e24
    #earth.px = -1*AU
    #earth.vy = 29.783 * 1000            # 29.783 km/sec
    earth.color = (88, 222, 249)
    earth.radius = 6731000

    moon = Body()
    moon.name = 'Moon'
    moon.mass = 7.36e22
    moon.px = 384_400_000 #407000000 / AU  #384_400_000 / AU
    print(f"speed of the moon => {384_400_000}")
    moon.vy = 1.022 * 1000
    moon.color = (154, 255, 0)
    moon.radius = 1736000

    rocket = Body()
    rocket.name = 'Rocket'
    rocket.mass = 500
    rocket.px = 0
    rocket.py = 6_731_000
    rocket.vy = 0
    rocket.color = (255, 0, 0)
    rocket.radius = 3 #5,6134
    
    rocket.thrust = 5000 # N

    #rocket.fuel_velocity = 100000
    #rocket.fuel_mass = 0.0001 # per second

    rocket.direction = [1, 0]
    
    space = gravity()

    space.angle = 90

    space.timestep = 1
    
    #space.loop([earth, rocket, sun], main=0, rocket=1)
    #space.loop([sun, earth], main=0, rocket=1, plot="Earth")

    space.loop([earth, rocket], main=0, rocket=1, plot_actions=True)

    #loop([sun, earth])
    #loop([earth, moon])

if __name__ == '__main__':
    main()