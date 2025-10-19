import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from quadruped_inverse import Leg
from quadruped_inverse import rotation_matrix

# Defining class for vizualizing robots's environment
class RobotEnvironment:

    def __init__(self):
        self.fig = plt.figure(figsize=(12, 11))                 # Creating figure window for plotting operations
        self.ax = self.fig.add_subplot(111, projection = '3d')  # Creating 3D coordinate
        self.ax.set_proj_type('ortho')                          # Orthagonal projection
        plt.subplots_adjust(bottom=0.3)                         # Offset for sliders 
        self.setup_axes()                                       
        
    # Method responsible for setting up the plotting environment
    def setup_axes(self):

        # Axes limits
        self.ax.set_xlim(-0.21, 0.21)
        self.ax.set_ylim(-0.21, 0.21)
        self.ax.set_zlim(-0.21, 0.21)

        # Axes labels
        self.ax.set_xlabel('X [m]', fontsize = 10)
        self.ax.set_ylabel('Y [m]', fontsize = 10)
        self.ax.set_zlabel('Z [m]', fontsize = 10)

        # Axes colors
        self.ax.xaxis.line.set_color("red")
        self.ax.yaxis.line.set_color("green")
        self.ax.zaxis.line.set_color("blue")

        # Surfaces colors
        self.ax.xaxis.pane.set_color((0.95, 0.95, 0.95, 0.1))
        self.ax.yaxis.pane.set_color((0.95, 0.95, 0.95, 0.1))
        self.ax.zaxis.pane.set_color((0.95, 0.95, 0.95, 0.1))

        # Grid on
        self.ax.grid(True)

        # Setting up title 
        self.ax.set_title("Quadruped robot", fontsize = 10)

        # Base view
        self.ax.view_init(elev=30, azim=15)

        self.ax.scatter([0], [0], [0], color='red', s=50, label='Center')

    # Method responsible for returning setupped environment
    def get_axes(self):
        return self.ax

# Class which represents the robot's body 
class RobotBody:

    def __init__(self):
        
        # Defining parameters of system
        self.length = Leg.body_length
        self.width = Leg.body_width
        self.high = Leg.body_high

    # Method responsible for creating robot corners
    def get_robot_corners(self, rotation = [0, 0, 0]):
        
        corners = np.array([
            [self.length / 2, self.width / 2, self.high / 2], [self.length / 2, self.width / 2, -self.high / 2],        # Left front
            [-self.length / 2, self.width / 2, self.high / 2], [-self.length / 2, self.width / 2, -self.high / 2],      # Left back
            [self.length / 2, -self.width / 2, self.high / 2], [self.length / 2, -self.width / 2, -self.high / 2],      # Right front
            [-self.length / 2, -self.width / 2, self.high / 2], [-self.length / 2, -self.width / 2, -self.high / 2],    # Right back
        ])
        
        # Place holder for corners after rotation
        corners_after_rotation = []

        # Rotation od robots corners
        for corner in corners:
            rotated_corner = rotation_matrix(rotation,'zyx') @ corner
            corners_after_rotation.append(rotated_corner)

        
        return np.array(corners_after_rotation)
    
    # Method for diagnostic checking of body geometry and rotation accuracy
    def diagnostic(self, corners_after_rotation, rotation = [0, 0, 0]):

    
        # Orthagonality test
        test_vector = np.array([1.0, 0.0, 0.0])
        rotated_vector = rotation_matrix([math.radians(30), 0, 0], 'xyz') @ test_vector
        rotation_error = np.linalg.norm(rotated_vector) - 1.0
        
        print(f"\nBasic test of rotation matrix:")
        print(f"Vector lenght error after rotation: {rotation_error:.10f} (powinno być ~0)")

        # Calculating lenght of edges
        x_edge = np.linalg.norm(corners_after_rotation[0] - corners_after_rotation[2])  # 0-2
        y_edge = np.linalg.norm(corners_after_rotation[0] - corners_after_rotation[4])  # 0-4 
        z_edge = np.linalg.norm(corners_after_rotation[0] - corners_after_rotation[1])  # 0-1

        # All X edges of robot's body
        x_edges = [
            np.linalg.norm(corners_after_rotation[0] - corners_after_rotation[2]),
            np.linalg.norm(corners_after_rotation[1] - corners_after_rotation[3]),
            np.linalg.norm(corners_after_rotation[4] - corners_after_rotation[6]),
            np.linalg.norm(corners_after_rotation[5] - corners_after_rotation[7])
            ]
        
        tolerance = 0.001  # Defined error tolerance
        
        # Printing diagnostic parameters
        print(f"\nWhole diagnostic of X edges:")
        for i, edge in enumerate(x_edges):
            print(f"Krawędź X-{i}: {edge:.6f} m (błąd: {edge-self.length:.6f} m)")

        print(f"\nEdge X summary:")
        print(f"Średnia długość krawędzi X: {np.mean(x_edges):.6f} m")
        print(f"Odchylenie standardowe: {np.std(x_edges):.6f} m")
        print(f"Maksymalny błąd: {max(x_edges)-self.length:.6f} m")

        print(f"\nY edge error: {y_edge - self.width:.6f} m")
        print(f"Z edge error: {z_edge - self.high:.6f} m")

# Class responsible for displaying robot
class RobotVisualization:

    # Method resposible for passing defined enviorment to robot's visualization class
    def __init__(self, environment):

        self.environment = environment
        self.axes = environment.get_axes()

    # Method responsible for displaying robot's body
    def draw_body(self, body, rotation = [0, 0, 0]): 

        # Passing robots body
        corners = body.get_robot_corners(rotation)

        # Combinations of points that form the edges of the robot body
        edges = [  
            [0, 2, 6, 4, 0],
            [1, 3, 7, 5, 1],
            [0, 1], [2, 3], [4, 5], [6, 7]
        ]

        # Plotting edges
        for edge in edges:
            x = [corners[i, 0] for i in edge]
            y = [corners[i, 1] for i in edge]
            z = [corners[i, 2] for i in edge]

            self.axes.plot(x, y, z, color = 'black', linewidth = 2)

    # Method resposible for drawing robot's legs
    def draw_leg(self, leg, target_pos_localy,  body_rotation=[0,0,0],):

        # Leg origin after rotation
        rotated_origin = rotation_matrix(body_rotation) @ leg.leg_origin
        
        # Calculate inverse kinematics of leg
        angles, joints = leg.inverse_kinematics(leg.bodyrotation_to_offset(target_pos_localy,body_rotation))
        
        # Transform joints to global coordinate
        global_joints = []
        
        for joint in joints:
            global_joint = rotation_matrix(body_rotation) @ joint 
            global_joint = global_joint + rotated_origin
            global_joints.append(global_joint)
        
        # Draw leg
        x = [joint[0] for joint in global_joints]
        y = [joint[1] for joint in global_joints]
        z = [joint[2] for joint in global_joints]
        
        self.axes.plot(x, y, z, 'r-', linewidth=3)
        self.axes.scatter(x, y, z, c='b', s=50)
       

            

if __name__ == "__main__":
   
    # Creating instances for visualization
    robot_environment = RobotEnvironment() 
    robot_body = RobotBody()
    visualization = RobotVisualization(robot_environment)
    legs = [Leg(i) for i in range(4)]

    # Defining end effectors positions of each leg with respect to local leg origin coordinate
    end_effectors_rel_to_leg_origin = [[0, 0.045, -0.21], [0, 0.045, -0.21],
                                        [0, -0.045, -0.21], [0, -0.045, -0.21]]

    # Defining length placing and dimmensions of sliders
    ax_roll = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_pitch = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_yaw = plt.axes([0.25, 0.1, 0.65, 0.03])
    
    # Creating sliders
    roll_slider = Slider(ax_roll, 'Roll (X)', -30, 30, valinit=0)
    pitch_slider = Slider(ax_pitch, 'Pitch (Y)', -30, 30, valinit=0)
    yaw_slider = Slider(ax_yaw, 'Yaw (Z)', -30, 30, valinit=0)


    def update(param):

        # Reading values from sliders
        roll = math.radians(round(roll_slider.val))
        pitch = math.radians(round(pitch_slider.val))
        yaw = math.radians(round(yaw_slider.val))

        
        rotation = [roll, pitch, yaw]

        # Clearing visualization
        visualization.axes.clear()
        # Re-creating enviorment
        visualization.environment.setup_axes()

        # Drawing robots visualization
        visualization.draw_body(robot_body, rotation)

        for i in range(4):
            visualization.draw_leg(legs[i], end_effectors_rel_to_leg_origin[i] , rotation)   

        visualization.environment.fig.canvas.draw_idle()  

    # Slider value value update with every differetince
    roll_slider.on_changed(update)
    pitch_slider.on_changed(update)
    yaw_slider.on_changed(update)

    update(None)

    plt.show()
    

  
