import math
import numpy as np

# Function which converts coordinates to angle in radians. 
# It ensures that the angle is not from Pi to -Pi but from 0 to 2Pi. 
# It is used to return correct angle values considering the signs of the arguments.
def coord_to_rad(x,y): 

    angle = math.atan2(y,x)
    if angle < 0:
        angle += 2*math.pi
    return angle

# Function that returns the rotation matrix with the possibility to change the order of rotations
def rotation_matrix(rotation = [0, 0, 0], order = 'zyx'): 

    roll, pitch, yaw = rotation[0], rotation[1], rotation[2]

    rot_X = np.array([[1,0,0],[0, math.cos(roll), -math.sin(roll)],[0, math.sin(roll), math.cos(roll)]])
    rot_Y = np.array([[math.cos(pitch), 0,math.sin(pitch)],[0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    rot_Z = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])

    if order == 'xyz' : return rot_X @ rot_Y @ rot_Z 
    elif order == 'xzy' : return rot_X @ rot_Z @ rot_Y 
    elif order == 'zxy' : return rot_Z @ rot_X @ rot_Y 
    elif order == 'yxz' : return rot_Y @ rot_X @ rot_Z 
    elif order == 'yzx' : return rot_Y @ rot_Z @ rot_X 
    elif order == 'zyx' : return rot_Z @ rot_Y @ rot_X 
    else: raise ValueError(f"Incorrect order:'{order}'")  

# Class which represents each leg
class Leg:

    # Defining physical parameters of the system (the same for each leg)
    body_length = 0.25 
    body_width = 0.10
    body_high = 0.05
    l1 = 0.045
    l2 = 0.15
    l3 = 0.15
    phi = math.pi/2

    def __init__(self, id): 

        self.id = id #Defining id of each leg

        # Assigning each leg to the correct place on the robot's body by ID number
        match self.id:
            #Left front leg
            case 0: 
                self.leg_origin = np.array([Leg.body_length/2, Leg.body_width / 2, 0])
                self.name = "Left front"
            #Left back leg
            case 1:
                self.leg_origin = np.array([-Leg.body_length/2, Leg.body_width / 2, 0])
                self.name = "Left back"
            #Right front
            case 2:
                self.leg_origin = np.array([Leg.body_length/2, -Leg.body_width / 2, 0])
                self.name = "Right front"
            #Right back
            case 3:
                self.leg_origin = np.array([-Leg.body_length/2, -Leg.body_width / 2, 0])
                self.name = "Right back"
            #Exeption value out of range 
            case _:
                raise ValueError("Leg id has to be from 0-3")

    # Method that rotates the robot's body coordinates to determine the end-effector's position after rotation
    # From global coordinates --> local rotated ones
    def bodyrotation_to_offset(self, xyz_endeffector_rel_2_leg_origin, rotation = [0,0,0]): 

        xyz_endeffector_rel_2_leg_origin = np.array(xyz_endeffector_rel_2_leg_origin)

        
        #It says us where is end effector in local robot coordinate after robot's body rotation
        xyz_endeffector_rel_2_body = xyz_endeffector_rel_2_leg_origin + self.leg_origin
        xyz_endeffector_rel_2_body_after_rotation = np.linalg.inv(rotation_matrix(rotation)) @ (xyz_endeffector_rel_2_body)

        #Vector offset in order to return to legs origin coordinate
        xyz_endeffector_rel_2_leg_origin =xyz_endeffector_rel_2_body_after_rotation - self.leg_origin
        return  xyz_endeffector_rel_2_leg_origin
    
    # Method responsible for calculating inverse kinematics of the robot's leg.
    # ARGUMENTS --> [X, Y, Z] (relative to leg origin), Returns --> angle combination that allows reaching the [X, Y, Z] position
    def inverse_kinematics(self, xyz):
        
        x, y, z = xyz[0], xyz[1], xyz[2]
        # Calculating length a, a --> from leg origin to end effector
        len_a = math.sqrt((y**2 + z**2))

        # Auxiliary angles to calculate theta1
        angle1 = coord_to_rad(y,z)
        angle2 = math.asin(Leg.l1 * (math.sin(Leg.phi) / len_a))
        angle3 = math.pi - Leg.phi - angle2

        # Differentiation between right and left side to find theta1 with respect to the same coordinate orientation
        if self.name in ["Left front", "Left back"]:
            theta1 = angle1 + angle3
        else:
            theta1 = angle1 - angle3

        # Providing theta1 between 0 - 2Pi
        if theta1 >=  2 * math.pi: theta1 = theta1 - 2 * math.pi

        # Differentiation between legs --> when robot's body rotates (e.g. along X axis), left legs have to rotate in the opposite direction to the right legs
        if self.name in ["Left front", "Left back"]:
            R = theta1 + self.phi - math.pi/2 #Left leg
        else:
            R = theta1 - self.phi - math.pi/2 #Right leg

        # Rotation matrix that allows us to project our view perpendicular to the XZ plane of the leg
        # From non-rotated leg origin --> rotated leg origin
        projection_rot = rotation_matrix([R,0,0]).T

        joint2 = np.array([0, Leg.l1 * math.cos(theta1), Leg.l1 * math.sin(theta1)]) # Vector from joint 1 to joint 2, relative to leg origin coordinates
        end_effector = np.array([x, y, z])
        end_effector_to_j2 = end_effector - joint2                                   # Vector from joint 2 to end effector
        end_effector_to_j2 = projection_rot @ end_effector_to_j2   # Above vector but taking into account rotated body coordinates

        x_,y_,z_ = end_effector_to_j2[0], end_effector_to_j2[1], end_effector_to_j2[2]

        #Length b --> from J2 to end effector
        len_b = math.sqrt((x_**2 + z_**2))

        # Safe condition to ensure that we don't exceed the range, which might lead to unexpected solutions
        if len_b > (Leg.l2 + Leg.l3):
            len_b = (Leg.l2 + Leg.l3) * 0.99
            raise ValueError("Position out of range")
        
        # Auxiliary angles to calculate theta2 and theta3
        beta1 = coord_to_rad(x_,z_)
        beta2 = math.acos((Leg.l2**2 + len_b**2 - Leg.l3**2) / (2 * Leg.l2 * len_b))
        beta3 = math.acos((Leg.l2**2 + Leg.l3**2 - len_b**2) / (2 * Leg.l2 * Leg.l3))

        theta2 = beta1 - beta2
        theta3 = math.pi - beta3

        # Array containing resulting angles
        angles = [theta1, theta2, theta3]

        # Placeholder array for all joint positions
        joint1 = np.array([0,0,0])

        if self.name in ["Left front", "Left back"]:
            # Calculating joint 3 position with respect to non-rotated robot body
            # From local rotated robot body coordinate --> non-rotated global coordinate
            joint3_not_rotated = np.array([Leg.l2 * math.cos(theta2), Leg.l1, Leg.l2 * math.sin(theta2)])
            joint3 = projection_rot.T @ joint3_not_rotated

            # Calculating end effector position with respect to rotated robot leg origin coordinate
            end_effector = joint3_not_rotated + np.array([Leg.l3 * math.cos(theta2 + theta3), 0, Leg.l3 * math.sin(theta2 + theta3)])
            end_effector = projection_rot.T @ end_effector
        else:
            # Calculating joint 3 position with respect to non-rotated robot body
            # From local rotated robot body coordinate --> non-rotated global coordinate
            joint3_not_rotated = np.array([Leg.l2 * math.cos(theta2), -Leg.l1, Leg.l2 * math.sin(theta2)])
            joint3 = np.linalg.inv(projection_rot) @ joint3_not_rotated

            # Calculating end effector position with respect to rotated robot leg origin coordinate
            end_effector = joint3_not_rotated + np.array([Leg.l3 * math.cos(theta2 + theta3), 0, Leg.l3 * math.sin(theta2 + theta3)])
            end_effector = np.linalg.inv(projection_rot) @ end_effector
        


        Joints = [joint1, joint2, joint3, end_effector]
        return [angles, Joints]
    
    # Returns leg origin taking into account body rotation
    def get_leg_origin(self, rotation = [0, 0, 0]):

        return rotation_matrix(rotation) @ self.leg_origin






# Testing module --> If you want to change resulting angles to fit your robot configuration, you can test your results here

if __name__  == "__main__":

    Test_leg1 = Leg(0)  #Left front
    Test_leg2 = Leg(1)  #Left back
    Test_leg3 = Leg(2)  #Right front
    Test_leg4 = Leg(3)  #Right back

    test_points = [
        [0.0, 0.045, -0.21],
        [0.0, -0.045, -0.21],
        [0.0, 0.2, -0.05]]
    
    angles, joints = Test_leg1.inverse_kinematics(Test_leg1.bodyrotation_to_offset(test_points[0], [math.radians(0), 0, 0]))
    
    print("\nLeft front:","Fi1: ",math.degrees(angles[0]),"Fi2: ",math.degrees(angles[1]), "Fi3: ", math.degrees(angles[2]),"\n")

    print("Joint 1:", joints[0], "Joint 2:", joints[1], "Joint 3:", joints[2],'\n',"Konc:",joints[3],'\n')

    
    
    