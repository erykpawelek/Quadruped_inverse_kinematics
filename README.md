# Quadruped Robot - Inverse Kinematics

Python implementation of inverse kinematics for a quadruped robot.

The program calculates compensation for each leg to enable the robot to rotate by a defined angle around a chosen axis.

## GUI and visualization view

![Default position](./Images/Gui_image.png)
![Rotated position](./Images/Gui_image.png)

## Equations and Diagrams

![Front view of the robot (left front leg)](./Images/Side%20movement%20calculation.png)
![Perpendicular view to the ZX plane of the leg (left front leg)](./Images/Perpandicular%20view%20to%20the%20leg.png)

We always calculate angles θ₂ and θ₃ by looking perpendicular to the leg. Therefore, we need to rotate our coordinate system by angle θ₁ to obtain this perpendicular view (see `quadruped_inverse.py`, line 114).
