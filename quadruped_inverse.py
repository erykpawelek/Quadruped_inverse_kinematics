import math
import numpy as np

#Function witch converts coordinate to angle in radians. It ensures that angle is not fom Pi to - Pi but from 0 to 2Pi it is used to return correct values of angle considering the signs of arguments.
def coord_to_rad(x,y): 

    angle = math.atan2(y,x)
    if angle < 0:
        angle += 2*math.pi
    return angle

#Function returns the rotation matrix when with posibility to change order of rotation
def rotation_matrix(rotation = [0, 0, 0], order = 'xyz'): 

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

#Class witch represents each leg
class Leg:

    #Defining phisical parameters of the system (for each leg all are same)
    body_length = 0.25 
    body_width = 0.10
    body_high = 0.05
    l1 = 0.045
    l2 = 0.15
    l3 = 0.15
    phi = math.pi/2

    def __init__(self, id): 

        self.id = id #Defining id of each leg

        #Assigning ech leg to the correct place of robot's body by id number
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

    #Method wich rotates the coordinate of robots body in order to find out in wich position will be leg's end_effector after the rotation
    #From Global coorginate --> local rotated one

    def bodyrotation_to_offset(self, xyz_endeffector_rel_2_leg_origin, rotation = [0,0,0]): 

        xyz_endeffector_rel_2_leg_origin = np.array(xyz_endeffector_rel_2_leg_origin)

        
        #It says us where is end effector in local robot coordinate after robot's body rotation
        xyz_endeffector_rel_2_body = xyz_endeffector_rel_2_leg_origin + self.leg_origin
        xyz_endeffector_rel_2_body_after_rotation = np.linalg.inv(rotation_matrix(rotation)) @ (xyz_endeffector_rel_2_body)

        #Vector offset in order to return to legs origin coordinate
        xyz_endeffector_rel_2_leg_origin =xyz_endeffector_rel_2_body_after_rotation - self.leg_origin
        return  xyz_endeffector_rel_2_leg_origin
    
    #Method resposible for calculating inverse kinematics of robot's leg.
    #  ARGUMENTS  --> [X, Y, Z](Relative to leg origin), Return --> angle combination of leg wich allows to reach the [X, Y, Z] position
    def inverse_kinematics(self, xyz):
        
        x, y, z = xyz[0], xyz[1], xyz[2]
        #Calculating legth a, a --> forom leg origin to end effector
        len_a = math.sqrt((y**2 + z**2))

        #Auxilary angles to calculate theta1
        angle1 = coord_to_rad(y,z)
        angle2 = math.asin(Leg.l1 * (math.sin(Leg.phi) / len_a))
        angle3 = math.pi - Leg.phi - angle2

        #Distinction beetwen right and left side in order to find out theta1 with respect to samely oriented coordinate
        if self.name in ["Left front", "Left back"]:
            theta1 = angle1 + angle3
        else:
            theta1 = angle1 - angle3

        #Prividing theta1 between 0 - 2Pi
        if theta1 >=  2 * math.pi: theta1 = theta1 - 2 * math.pi

        #Distinction beetwen legs --> 
        # when robot's body rotate f.ex. along X with positive angle left legs hawe to ratate in contrary angle to right legs (arm rotation theta1)
        if self.name in ["Left front", "Left back"]:
            R = theta1 + self.phi - math.pi/2 #Left leg
        else:
            R = theta1 - self.phi - math.pi/2 #Right leg

        #Rotaion matrix wich allows us to to project our wiev parpendicular to XZ plane of the leg
        # From non roatadet leg origin --> rotated leg origin
        projection_rot =np.linalg.inv(rotation_matrix([R,0,0]))

        joint2 = np.array([0, Leg.l1 * math.cos(theta1), Leg.l1 * math.sin(theta1)]) #Vector from J1 to J2, relative to leg origin coordinate
        end_effector = np.array([x, y, z])
        end_effector_to_j2 = end_effector - joint2                                   #Vector from J2 to end effector
        end_effector_to_j2 = projection_rot @ end_effector_to_j2   #Above vector but taking into account rotated body coordiante

        x_,y_,z_ = end_effector_to_j2[0], end_effector_to_j2[1], end_effector_to_j2[2]

        #Length b --> from J2 to end effector
        len_b = math.sqrt((x_**2 + z_**2))

        #Sefe condition to ensure that we dont reach out of our range ,which might get us inpredicted solutions
        if len_b > (Leg.l2 + Leg.l3):
            len_b = (Leg.l2 + Leg.l3) * 0.99
            raise ValueError("Position out of range")
        
        #Auxilary angles to calculate theta2 and theta3
        beta1 = coord_to_rad(x_,z_)
        beta2 = math.acos((Leg.l2**2 + len_b**2 - Leg.l3**2) / (2 * Leg.l2 * len_b))
        beta3 = math.acos((Leg.l2**2 + Leg.l3**2 - len_b**2) / (2 * Leg.l2 * Leg.l3))

        theta2 = beta1 - beta2
        theta3 = math.pi - beta3

        #Array containing resulting angles
        angles = [theta1, theta2, theta3]

        #Placeholder array for all joints positions 
        joint1 = np.array([0,0,0])

        if self.name in ["Left front", "Left back"]:
           #Calculating joint 3 position with respect to non rotated robots body 
            #From local rotated robots body coorinate --> non rotated global coordinate
            joint3_not_rotated = np.array([Leg.l2 * math.cos(theta2), Leg.l1, Leg.l2 * math.sin(theta2)])
            joint3 = np.linalg.inv(projection_rot) @ joint3_not_rotated

            #Calculating end_effector position with respect to roatated robots leg origin coordinate
            end_effector = joint3_not_rotated + np.array([Leg.l3 * math.cos(theta2 + theta3), 0, Leg.l3 * math.sin(theta2 + theta3)])
            end_effector = np.linalg.inv(projection_rot) @ end_effector
        else:
           #Calculating joint 3 position with respect to non rotated robots body 
            #From local rotated robots body coorinate --> non rotated global coordinate
            joint3_not_rotated = np.array([Leg.l2 * math.cos(theta2), -Leg.l1, Leg.l2 * math.sin(theta2)])
            joint3 = np.linalg.inv(projection_rot) @ joint3_not_rotated

            #Calculating end_effector position with respect to roatated robots leg origin coordinate
            end_effector = joint3_not_rotated + np.array([Leg.l3 * math.cos(theta2 + theta3), 0, Leg.l3 * math.sin(theta2 + theta3)])
            end_effector = np.linalg.inv(projection_rot) @ end_effector
        


        Joints = [joint1, joint2, joint3, end_effector]
        return [angles, Joints]
    
    def get_leg_origin(self, rotation = [0, 0, 0]):

        return rotation_matrix(rotation) @ self.leg_origin




#Diafnostic module


if __name__  == "__main__":

    Test_leg1 = Leg(0)  #Left front
    Test_leg2 = Leg(1)  #Left back
    Test_leg3 = Leg(2)  #Right front
    Test_leg4 = Leg(3)  #Right back
    test_points = [
        [0.0, 0.045, -0.21],
        [0.0, -0.045, -0.21],
        [0.0, 0.2, -0.05]]
    # test_ = Test_leg1.inverse_kinematics(Test_leg1.bodyrotation_to_offset(test_points[0], [math.radians(0), 0, 0]))
    # macierz_obrotu = rotation_matrix([math.radians(0), 0, 0])
    # leg_origin = Test_leg1.get_leg_origin([math.radians(0), 0, 0])
    # angles = test_[0]
    # joints = test_[1]
    # print("Lewa przednia:","Fi1: ",math.degrees(angles[0]),"Fi2: ",math.degrees(angles[1]), "Fi3: ", math.degrees(angles[2]),"\n")
    # print("Joint 1:", joints[0], "Joint 2:", joints[1], "Joint 3:", joints[2],'\n',"Koncowka:",joints[3],'\n')
    # print("Wspolrzedne leg origin: ", "X: ", leg_origin[0], "Y: ", leg_origin[1], "Z: ", leg_origin[2])
    # print("macierz obrotu: ", macierz_obrotu)

    # test_angles = Test_leg2.inverse_kinematics(Test_leg2.bodyrotation_to_offset(test_points[0]))
    # print("Lewa tył:","Fi1: ",math.degrees(test_angles[0]),"Fi2: ",math.degrees(test_angles[1]), "Fi3: ", math.degrees(test_angles[2]))
    # test_angles = Test_leg3.inverse_kinematics(Test_leg3.bodyrotation_to_offset(test_points[1]))
    # print("Prawa przednia:","Fi1: ",math.degrees(test_angles[0]),"Fi2: ",math.degrees(test_angles[1]), "Fi3: ", math.degrees(test_angles[2]))
    angles, joints = Test_leg3.inverse_kinematics(Test_leg4.bodyrotation_to_offset(test_points[1],[math.radians(0), 0, 0] ))
    print("Prawa przednia:","Fi1: ",math.degrees(angles[0]),"Fi2: ",math.degrees(angles[1]), "Fi3: ", math.degrees(angles[2]))
    print("Joint 1:", joints[0], "Joint 2:", joints[1], "Joint 3:", joints[2],'\n',"Koncowka:",joints[3],'\n')    