import rosbag2_py
import numpy as np
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
import tf_transformations  # To handle quaternion to euler conversion
from visualization_msgs.msg import Marker
from scipy.interpolate import interp1d
from geometry_msgs.msg import PointStamped

def extract_estimate_positions_ros2(bag_path, topic_name):
    positions = []  # List to store x, y positions
    timestamps = []  # List to store timestamps

    # Open the ROS 2 bag
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get all topics and types
    topic_types = reader.get_all_topics_and_types()
    topic_type_dict = {topic.name: topic.type for topic in topic_types}

    # Check if the topic exists in the bag file
    if topic_name not in topic_type_dict:
        raise ValueError(f"Topic '{topic_name}' not found in the bag file.")

    # Read messages
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == topic_name:
            # Deserialize data using the PointStamped message type
            msg = deserialize_message(data, PointStamped)
            x = msg.point.x
            y = msg.point.y
            timestamp_sec = t / 1e9  # Convert from nanoseconds to seconds

            # Append position and timestamp
            positions.append((x, y))  # Ensure positions is a list of tuples
            timestamps.append(timestamp_sec)

    if not positions:
        print(f"No data extracted from topic {topic_name}.")
    else:
        print(f"Extracted {len(positions)} positions from topic {topic_name}.")
        
    return timestamps, positions

# Function to extract positions from an Odometry topic
def extract_odom_positions_ros2(bag_path, topic_name):
    positions = []
    timestamps = []

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    topic_type_dict = {topic.name: topic.type for topic in topic_types}

    if topic_name not in topic_type_dict:
        raise ValueError(f"Topic '{topic_name}' not found in the bag file.")

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == topic_name:
            msg = deserialize_message(data, Odometry)
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            quat = msg.pose.pose.orientation
            _, _, theta = tf_transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

            timestamp_sec = t / 1e9
            positions.append((x, y, theta))
            timestamps.append(timestamp_sec)

    if not positions:
        print(f"No data extracted from topic {topic_name}.")
    else:
        print(f"Extracted {len(positions)} positions from topic {topic_name}.")
        
    return timestamps, positions

# Function to extract particle data and compute the mean trajectory
def extract_particles_ros2(bag_path, topic_name):
    timestamps = []  # List to store timestamps
    particles_per_timestep = []  # List to store particles at each timestamp

    # Open the ROS 2 bag
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get all topics and types
    topic_types = reader.get_all_topics_and_types()
    topic_type_dict = {topic.name: topic.type for topic in topic_types}

    # Check if the topic exists in the bag file
    if topic_name not in topic_type_dict:
        raise ValueError(f"Topic '{topic_name}' not found in the bag file.")

    # Read messages
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == topic_name:
            # Deserialize data using the Marker message type
            msg = deserialize_message(data, Marker)

            # Extract particle positions
            particles = [(point.x, point.y) for point in msg.points]
            timestamp_sec = t / 1e9  # Convert from nanoseconds to seconds

            # Append the extracted particles and timestamp
            timestamps.append(timestamp_sec)
            particles_per_timestep.append(particles)

    if not particles_per_timestep:
        print(f"No particle data extracted from topic {topic_name}.")
    else:
        print(f"Extracted {len(particles_per_timestep)} timesteps from topic {topic_name}.")
        
    return timestamps, particles_per_timestep

# Function to interpolate data to real timestamps
def interpolate_data(timestamps_real, timestamps_source, source_positions):
    # Extract x, y, and theta
    source_x, source_y, source_theta = zip(*source_positions)

    # Create interpolation functions for x, y, and theta
    interp_x = interp1d(timestamps_source, source_x, kind='linear', fill_value="extrapolate")
    interp_y = interp1d(timestamps_source, source_y, kind='linear', fill_value="extrapolate")
    interp_theta = interp1d(timestamps_source, source_theta, kind='linear', fill_value="extrapolate")

    # Interpolate the source data at real timestamps
    x_interp = interp_x(timestamps_real)
    y_interp = interp_y(timestamps_real)
    theta_interp = interp_theta(timestamps_real)

    # Return interpolated data
    return list(zip(x_interp, y_interp, theta_interp))


def plot_trajectories_with_particles_and_estimate(
    real_positions,
    pf_positions,
    landmarks_ids,
    landmarks_x,
    landmarks_y,
    #ground_truth_positions,
    estimate_positions,
    particles_per_timestep,
    particle_timestamps,
):
    real_x, real_y, _ = zip(*real_positions)
    pf_x, pf_y, _ = zip(*pf_positions)
    #ground_truth_x, ground_truth_y, _ = zip(*ground_truth_positions)

    estimate_x = [pos[0] for pos in estimate_positions]
    estimate_y = [pos[1] for pos in estimate_positions]

    plt.figure(figsize=(10, 8))

    # Plot ground truth trajectory
    #plt.plot(ground_truth_x, ground_truth_y, label='/ground_truth Trajectory', color='green', linewidth=1.5)

    # Plot landmarks
    #plt.scatter(landmarks_x, landmarks_y, label='Landmarks', color='gray', marker='|', s=200)

    # Plot particles
    for i, (timestamp, particles) in enumerate(zip(particle_timestamps, particles_per_timestep)):
        particle_x = [p[0] for p in particles]
        particle_y = [p[1] for p in particles]

        if i == 0:
            # Initial particles: lighter and transparent
            plt.scatter(particle_x, particle_y, color='gray', alpha=0.1, s=2, label='Initial Particles')
        elif i >= 8:
            # Start plotting particles from the 10th step
            plt.scatter(particle_x, particle_y, color='red', alpha=0.4, s=2)

    # Plot estimate positions
    #plt.scatter(estimate_x, estimate_y, label='/estimate Positions', color='red', s=10, marker='o')

    # Plot real trajectory
    plt.plot(real_x, real_y, label='/odom Trajectory', color='green', linewidth=1.5)

    # Plot PF trajectory
    plt.plot(pf_x, pf_y, label='/pf Trajectory', color='blue',  linewidth=1.5)

    # Finalize plot
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')

    plt.title('Trajectories with Particle Evolution and Estimate Position')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# Function to plot x, y, and theta over time
def plot_x_y_theta(timestamps_real, real_positions, timestamps_pf, pf_positions):#, timestamps_ground_truth, ground_truth_positions):
    real_x, real_y, real_theta = zip(*real_positions)
    pf_x, pf_y, pf_theta = zip(*pf_positions)
    #ground_truth_x, ground_truth_y, ground_truth_theta = zip(*ground_truth_positions)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Plot X position over time
    axs[0].plot(timestamps_pf, pf_x, label='/ekf/pose/pose/position/x', color='blue', linestyle='-', linewidth=1)
    axs[0].plot(timestamps_real, real_x, label='/odom/pose/pose/position/x', color='red', linestyle='-', linewidth=1)
    #axs[0].plot(timestamps_ground_truth, ground_truth_x, label='/ground_truth/pose/pose/position/x', color='green', linestyle=':', linewidth=1)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(loc='upper right')
    axs[0].set_title('X Position Over Time')
    axs[0].set_ylabel('X (m)')
    axs[0].set_xlabel('Time (s)')

    # Plot Y position over time
    axs[1].plot(timestamps_pf, pf_y, label='/ekf/pose/pose/position/y', color='blue', linestyle='-', linewidth=1)
    axs[1].plot(timestamps_real, real_y, label='/odom/pose/pose/position/y', color='red', linestyle='-', linewidth=1)
    #axs[1].plot(timestamps_ground_truth, ground_truth_y, label='/ground_truth/pose/pose/position/y', color='green', linestyle=':', linewidth=1)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend(loc='upper right')
    axs[1].set_title('Y Position Over Time')
    axs[1].set_ylabel('Y (m)')
    axs[1].set_xlabel('Time (s)')

    # Plot Theta (orientation) over time
    axs[2].plot(timestamps_pf, pf_theta, label='/ekf/pose/orientation/yaw', color='blue', linestyle='-', linewidth=1)
    axs[2].plot(timestamps_real, real_theta, label='/odom/pose/orientation/yaw', color='red', linestyle='-', linewidth=1)
    #axs[2].plot(timestamps_ground_truth, ground_truth_theta, label='/ground_truth/pose/orientation/yaw', color='green', linestyle=':', linewidth=1)
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend(loc='upper right')
    axs[2].set_title('Theta (Orientation) Over Time')
    axs[2].set_ylabel('Theta (rad)')
    axs[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()


# Function to compute MAE and RMSE
def compute_error(real_positions, deformed_positions):
    real_x, real_y, real_theta = zip(*real_positions)
    deformed_x, deformed_y, deformed_theta = zip(*deformed_positions)

    # Compute MAE and RMSE for X, Y, and Theta
    mae_x = np.mean(np.abs(np.array(real_x) - np.array(deformed_x)))
    rmse_x = np.sqrt(np.mean((np.array(real_x) - np.array(deformed_x)) ** 2))

    mae_y = np.mean(np.abs(np.array(real_y) - np.array(deformed_y)))
    rmse_y = np.sqrt(np.mean((np.array(real_y) - np.array(deformed_y)) ** 2))

    mae_theta = np.mean(np.abs(np.array(real_theta) - np.array(deformed_theta)))
    rmse_theta = np.sqrt(np.mean((np.array(real_theta) - np.array(deformed_theta)) ** 2))

    return {
        "MAE_X": mae_x, "RMSE_X": rmse_x,
        "MAE_Y": mae_y, "RMSE_Y": rmse_y,
        "MAE_Theta": mae_theta, "RMSE_Theta": rmse_theta
    }

# Main execution
bag_path = "/home/francesco-masin/bag_files/realbag179"

# New landmarks with IDs and coordinates
landmarks_ids = [0,1,2,3,4,5]
landmarks_x = [1.80, -0.45, -0.2, 1.20, 1.33, -0.09]
landmarks_y = [0.14, -0.11, 1.76, 1.18, -1.59, -1.64]

# Extract data from bag file
timestamps_real, real_positions = extract_odom_positions_ros2(bag_path, "/odom")
timestamps_pf, pf_positions = extract_odom_positions_ros2(bag_path, "/pf")
#timestamps_ground_truth, ground_truth_positions = extract_odom_positions_ros2(bag_path, "/ground_truth")
timestamps_estimate, estimate_positions = extract_estimate_positions_ros2(bag_path, "/estimate")

# Extract particles
particle_timestamps, particles_per_timestep = extract_particles_ros2(bag_path, "/particles")

if real_positions and pf_positions: #and ground_truth_positions:
    interpolated_pf_positions = interpolate_data(timestamps_real, timestamps_pf, pf_positions)
    #interpolated_ground_truth_positions = interpolate_data(timestamps_real, timestamps_ground_truth, ground_truth_positions)

    errors_pf = compute_error(real_positions, interpolated_pf_positions)
    print("Errors between /odom and /pf:")
    for key, value in errors_pf.items():
        print(f"{key}: {value}")

    # Plot trajectories, particles, and estimate positions
    plot_trajectories_with_particles_and_estimate(
        real_positions,
        interpolated_pf_positions,
        landmarks_ids,
        landmarks_x,
        landmarks_y,
        #interpolated_ground_truth_positions,
        estimate_positions,
        particles_per_timestep,
        particle_timestamps,
    )
    plot_x_y_theta(
        timestamps_real, 
        real_positions, 
        timestamps_real, 
        interpolated_pf_positions, 
        #timestamps_ground_truth, 
        #interpolated_ground_truth_positions
        )


 

