import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class CelestialBody:
    def __init__(self, name, mass, position, velocity):
        self.name = name
        self.mass = mass
        self.position = position
        self.velocity = velocity

    def update_velocity(self, force, dt):
        acceleration = [f / self.mass for f in force]
        self.velocity = [v + a * dt for v, a in zip(self.velocity, acceleration)]

    def update_position(self, dt):
        self.position = [p + v * dt for p, v in zip(self.position, self.velocity)]

def calculate_gravitational_force(body1, body2):
    G = 6.67430e-11
    distance = calculate_distance(body1.position, body2.position)
    magnitude = (G * body1.mass * body2.mass) / (distance ** 2)
    direction = calculate_direction(body1.position, body2.position)
    force = [magnitude * d for d in direction]
    return force

def calculate_distance(position1, position2):
    return math.sqrt(sum((p2 - p1) ** 2 for p1, p2 in zip(position1, position2)))

def calculate_direction(position1, position2):
    distance = calculate_distance(position1, position2)
    direction = [(p2 - p1) / distance for p1, p2 in zip(position1, position2)]
    return direction

def add_orbiting_body(celestial_bodies):
    if len(celestial_bodies) < 1:
        print("No existing celestial bodies to orbit.")
        return

    orbiting_body_index = int(input("Enter the index of the existing celestial body to orbit: "))
    if orbiting_body_index < 0 or orbiting_body_index >= len(celestial_bodies):
        print("Invalid celestial body index.")
        return

    orbiting_body = celestial_bodies[orbiting_body_index]
    radius_ratio = float(input("Enter the length ratio of orbit radius: "))
    radius_unit = input("Enter the unit for radius (light year/AU/km/radius of the Milky Way): ")
    radius=convert_distance_to_m(radius_ratio, radius_unit)

    x = orbiting_body.position[0]-radius
    y = orbiting_body.position[1]
    z = orbiting_body.position[2]
    position = [x, y, z]

    name = input("Enter the name of the orbiting celestial body: ")
    mass_ratio = float(input("Enter the mass ratio of the orbiting celestial body: "))
    mass_unit = input("Enter the mass unit (sun/earth/jupiter): ")
    velocity = math.sqrt(orbiting_body.mass * 6.67430e-11 / radius)
    velocity *= convert_distance_to_m(1, "km") / convert_distance_to_m(1, "km")

    celestial_bodies.append(CelestialBody(name, convert_mass_to_kg(mass_ratio, mass_unit), position, [0.0, velocity, 0.0]))

def simulate_motion(celestial_bodies, dt, steps, merge_distance):
    positions = [[] for _ in range(len(celestial_bodies))]

    for _ in range(steps):
        for i in range(len(celestial_bodies)-1):
            body1 = celestial_bodies[i]

            for j in range(i+1, len(celestial_bodies)):
                
                body2 = celestial_bodies[j]

                force1 = calculate_gravitational_force(body1, body2)
                force2 = calculate_gravitational_force(body2, body1)

                body1.acceleration = [f / body1.mass for f in force1]
                body2.acceleration = [f / body2.mass for f in force2]

            body1.update_velocity(force1,dt)
            body1.update_position(dt)
            body2.update_velocity(force2,dt)
            body2.update_position(dt)
            positions[i].append(body1.position.copy())
            positions[j].append(body2.position.copy())

        celestial_bodies = merge_collided_bodies(celestial_bodies, merge_distance)
    plot_celestial_bodies(positions)


def merge_bodies(body1,body2):
    velocity=[]
    mass= body1.mass+body2.mass
    position=body1.position
    for i in range (3):
        velocity[i]=(body1.mass* body1.velocity[i]+ body2.mass* body2.velocity[i])/mass
    return CelestialBody("merged_mody", mass, position, velocity)
    
def merge_collided_bodies(celestial_bodies, merge_distance):
    merged_bodies = []

    for i in range(len(celestial_bodies)):
        body1 = celestial_bodies[i]

        if body1 not in merged_bodies:
            for j in range(i + 1, len(celestial_bodies)):
                body2 = celestial_bodies[j]

                if body2 not in merged_bodies:
                    distance = calculate_distance(body1.position, body2.position)

                    if distance <= merge_distance:
                        merged_body = merge_bodies(body1, body2)
                        merged_bodies.extend([body1, body2])
                        celestial_bodies[i] = merged_body

    celestial_bodies = [body for body in celestial_bodies if body not in merged_bodies]

    return celestial_bodies

def plot_celestial_bodies(positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, body_positions in enumerate(positions):
        x = [p[0]  for p in body_positions]
        y = [p[1]  for p in body_positions]
        z = [p[2]  for p in body_positions]

        ax.scatter(x, y, z, label=f'Body {i + 1}',s=1)

   
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    

    ax.legend()
    plt.show()

def list_celestial_bodies(celestial_bodies):
    print("Current Celestial Bodies:")
    for i, body in enumerate(celestial_bodies):
        print(f"Body {i + 1}:")
        print(f"  Name: {body.name}")
        print(f"  Mass: {body.mass} kg")
        print(f"  Position: {body.position} meters")
        print(f"  Velocity: {body.velocity} m/s")
        print()

def convert_mass_to_kg(mass_ratio, mass_unit):
    mass_units = {
        'sun': 1.989 * (10 ** 30),
        'earth': 5.972 * (10 ** 24),
        'jupiter': 1.898 * (10 ** 27),
    }
    return mass_ratio * mass_units[mass_unit]

def convert_distance_to_m(distance, distance_unit):
    distance_units = {
        'light year': 9.461 * (10 ** 15),
        'AU': 1.496 * (10 ** 11),
        'km': 1000,
        'radius of the Milky Way': 5.5 * (10 ** 20)
    }
    return distance * distance_units[distance_unit]

def main():
    celestial_bodies = []

    while True:
        print("Select an option:")
        print("1. Add a still celestial body")
        print("2. Add a celestial body orbiting an existing celestial body")
        print("3. Shoot a celestial body")
        print("4. Simulate motion")
        print("5. List current celestial bodies")
        print("6. Quit")

        choice = int(input("Enter your choice: "))

        if choice == 1:
            name = input("Enter the name of the celestial body: ")
            mass_ratio = float(input("Enter the mass ratio (x): "))
            mass_unit = input("Enter the mass unit (sun/earth/jupiter): ")
            x = float(input("Enter the x-coordinate: "))
            y = float(input("Enter the y-coordinate: "))
            z = float(input("Enter the z-coordinate: "))
            position_unit = input("Enter the unit for position (light year/AU/km/radius of the Milky Way): ")
            position = [
                convert_distance_to_m(x, position_unit),
                convert_distance_to_m(y, position_unit),
                convert_distance_to_m(z, position_unit)
            ]
            mass = convert_mass_to_kg(mass_ratio, mass_unit)

            celestial_bodies.append(CelestialBody(name, mass, position, [0.0, 0.0, 0.0]))

        elif choice == 2:
            add_orbiting_body(celestial_bodies)

        elif choice == 3:
            name = input("Enter the name of the celestial body: ")
            mass_ratio = float(input("Enter the mass ratio (x): "))
            mass_unit = input("Enter the mass unit (sun/earth/jupiter): ")
            mass = convert_mass_to_kg(mass_ratio, mass_unit)
            x = float(input("Enter the x-coordinate: "))
            y = float(input("Enter the y-coordinate: "))
            z = float(input("Enter the z-coordinate: "))
            position_unit = input("Enter the unit for position (light year/AU/km/radius of the Milky Way): ")
            position = [
                convert_distance_to_m(x, position_unit),
                convert_distance_to_m(y, position_unit),
                convert_distance_to_m(z, position_unit)
            ]
            x = float(input("Enter the x-coordinate of direction: "))
            y = float(input("Enter the y-coordinate of direction: "))
            z = float(input("Enter the z-coordinate of direction: "))
            direction=[x, y, z]
            m=math.sqrt(x**2 + y**2 + z**2)
            
            velocity_magnitude = float(input("Enter the velocity magnitude: "))
            velocity = [velocity_magnitude * direction[0]/m, velocity_magnitude * direction[1]/m, velocity_magnitude * direction[2]/m]

            celestial_body = CelestialBody(name, mass, position, velocity)
            celestial_bodies.append(celestial_body)

            print("Celestial body added successfully!")
 
        elif choice == 4:
            dt = float(input("Enter the time step size (in seconds): "))
            steps = int(input("Enter the total steps: "))
            merge_distance = float(input("Enter the merge distance (in meters): "))

            simulate_motion(celestial_bodies, dt, steps, merge_distance)
            print("Simulation completed.")

        elif choice == 5:
            list_celestial_bodies(celestial_bodies)

        elif choice == 6:
            print("Quitting the program.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
