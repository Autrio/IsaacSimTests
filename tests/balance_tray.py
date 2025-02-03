import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.spatial.transform import Rotation as R
from icecream import ic

# Load the model and create a simulation
model = mujoco.MjModel.from_xml_path('./tray_viz.xml')
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)

# Desired position for the cylinder
desired_pos = np.array([0.0, 0.0, 0.0])

# Proportional and derivative control gains
Kp = -1.5
Kd = -0.2

# Lists to store error values for y and x directions
errors_y = []
errors_x = []
# time.sleep(2)
# Initialize previous errors for derivative calculation
prev_error_y = 0.0
prev_error_x = 0.0
viewer.cam.distance = 3
start_time = time.time()
times = []  




while viewer.is_running():
    # Get the current position of the cylinder
    cylinder_pos = data.body("cylinder_body").xpos

    # Calculate the error in the y and x directions
    error_y = desired_pos[1] - cylinder_pos[1]
    error_x = cylinder_pos[0] - desired_pos[0]

    # Store the error magnitudes
    errors_y.append(np.linalg.norm(error_y))
    errors_x.append(np.linalg.norm(error_x))

    # Calculate the derivatives of the errors
    error_derivative_y = (error_y - prev_error_y) / 0.002  # Assuming time step of 0.002 seconds
    error_derivative_x = (error_x - prev_error_x) / 0.002

    # Calculate the control inputs (angles of the tray)
    tray_angle_y = Kp * error_y + Kd * error_derivative_y  # PD control in y-direction
    tray_angle_x = Kp * error_x + Kd * error_derivative_x  # PD control in x-direction

    # Apply the control inputs
    data.ctrl[0] = tray_angle_y
    data.ctrl[1] = tray_angle_x

    quat = [data.body("collision_object").xquat[1], data.body("collision_object").xquat[2], data.body("collision_object").xquat[3], data.body("collision_object").xquat[0]]
    rotation = np.linalg.inv(R.as_matrix(R.from_quat(quat)))
    new_quat = R.from_quat(quat).as_matrix() @ R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
    new_quat = R.from_matrix(new_quat).as_quat()
    new_quat = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
    translation = data.body("collision_object").xpos +  np.array([model.geom(model.body("collision_object").geomadr).size[0], 0 , 0]) @ rotation
    # Step the simulation
    data.mocap_pos[0] = translation
    radial_dist = np.linalg.norm(data.body("collision_object").xpos - data.mocap_pos[0])
    data.mocap_quat[0] = new_quat
    mujoco.mj_step(model, data)
    
    

    # Render the simulation
    viewer.sync()
    time.sleep(0.002)
    # Update previous errors
    prev_error_y = error_y
    prev_error_x = error_x
    times.append(time.time()-start_time)
    start_time = time.time()

# Plot the error over time for y and x directions
