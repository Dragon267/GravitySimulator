import tensorflow as tf
from BOT_SIMULATOR import BOT_SIMULATOR
from Body import Body

model = tf.keras.models.load_model('/Users/danila/Documents/space_flight_simulator/rocket_alpha_1.model')

def main():
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
    rocket.px = 6_731_000
    rocket.vy = 0
    rocket.color = (255, 0, 0)
    rocket.radius = 3 #5,6134
    
    rocket.thrust = 5000 # N

    #rocket.fuel_velocity = 100000
    #rocket.fuel_mass = 0.0001 # per second

    rocket.direction = [1, 0]
    
    space = BOT_SIMULATOR()

    space.angle = 90

    space.timestep = 1
    
    #space.loop([earth, rocket, sun], main=0, rocket=1)
    #space.loop([sun, earth], main=0, rocket=1, plot="Earth")

    space.loop([earth, rocket], model, main=0, rocket=1, plot_actions=True)

    #

if __name__ == '__main__':
    main()