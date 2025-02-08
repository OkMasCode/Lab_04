import rosbag2_py
import numpy as np
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
import tf_transformations  # To handle quaternion to euler conversion
from scipy.interpolate import interp1d
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Function to extract x, y, and theta positions from any Odometry topic in a ROS 2 bag
def extract_odom_positions_ros2(bag_path, topic_name):
    positions = []  # List to store x, y, and theta positions
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
            # Deserialize data using the Odometry message type
            msg = deserialize_message(data, Odometry)
            
            # Extract position and orientation (quaternion)
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            quat = msg.pose.pose.orientation
            # Convert quaternion to Euler angles (yaw, pitch, roll)
            _, _, theta = tf_transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

            # Append the extracted position and orientation along with timestamp
            timestamp_sec = t / 1e9  # Convert from nanoseconds to seconds
            positions.append((x, y, theta))
            timestamps.append(timestamp_sec)

    if not positions:
        print(f"No data extracted from topic {topic_name}.")
    else:
        print(f"Extracted {len(positions)} positions from topic {topic_name}.")
        
    return timestamps, positions

def extract_path_ros2(bag_path, topic_name):
    positions = []  # List to store x, y, and theta positions
    
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
            # Deserialize data using the Odometry message type
            msg = deserialize_message(data, Odometry)
            
            # Extract position and orientation (quaternion)
            x = msg.poses.pose.position.x
            y = msg.poses.pose.position.y

            # Append the extracted position and orientation along with timestamp
            timestamp_sec = t / 1e9  # Convert from nanoseconds to seconds
            positions.append((x, y))

    if not positions:
        print(f"No data extracted from topic {topic_name}.")
    else:
        print(f"Extracted {len(positions)} positions from topic {topic_name}.")
        
    return positions



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


# Function to plot the trajectories with landmark IDs
def plot_trajectories(real_positions, path_positions, landmark_ids, landmarks_x, landmarks_y):
    real_x, real_y, _ = zip(*real_positions)
    path_x, path_y, _ = zip(*path_positions)
    #ground_truth_x, ground_truth_y, _ = zip(*ground_truth_positions)

    plt.figure(figsize=(10, 8))

    # Plot the real trajectory
    plt.plot(real_x, real_y, label='/ground_truth', color='red', linestyle='-', linewidth=1)

    # Plot the EKF trajectory
    plt.plot(path_x, path_y, label='/A* trajectory', color='blue', linestyle='-', linewidth=1)

    # Plot landmarks
    plt.scatter(landmarks_x, landmarks_y, label='Landmarks', color='gray', marker='o', s=250)

    # Annotate each landmark with its ID
    for i, (x, y, landmarks_id) in enumerate(zip(landmarks_x, landmarks_y, landmark_ids)):
        plt.text(x+0.05, y+0.05, f'Landmark {str(landmarks_id)}', fontsize=8, color='gray')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')

    plt.title('Trajectory Comparison')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

"""
# Function to plot x, y, and theta over time
def plot_x_y_theta(timestamps_real, real_positions, timestamps_path, path_positions):
    real_x, real_y, real_theta = zip(*real_positions)
    ekf_x, ekf_y, ekf_theta = zip(*ekf_positions)
    #ground_truth_x, ground_truth_y, ground_truth_theta = zip(*ground_truth_positions)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Plot X position over time
    axs[0].plot(timestamps_ekf, ekf_x, label='/ekf/pose/pose/position/x', color='blue', linestyle='-', linewidth=1)
    axs[0].plot(timestamps_real, real_x, label='/odom/pose/pose/position/x', color='red', linestyle='-', linewidth=1)
    #axs[0].plot(timestamps_ground_truth, ground_truth_x, label='/ground_truth/pose/pose/position/x', color='green', linestyle=':', linewidth=1)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(loc='upper right')
    axs[0].set_title('X Position Over Time')
    axs[0].set_ylabel('X (m)')
    axs[0].set_xlabel('Time (s)')

    # Plot Y position over time
    axs[1].plot(timestamps_ekf, ekf_y, label='/ekf/pose/pose/position/y', color='blue', linestyle='-', linewidth=1)
    axs[1].plot(timestamps_real, real_y, label='/odom/pose/pose/position/y', color='red', linestyle='-', linewidth=1)
    #axs[1].plot(timestamps_ground_truth, ground_truth_y, label='/ground_truth/pose/pose/position/y', color='green', linestyle=':', linewidth=1)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend(loc='upper right')
    axs[1].set_title('Y Position Over Time')
    axs[1].set_ylabel('Y (m)')
    axs[1].set_xlabel('Time (s)')

    # Plot Theta (orientation) over time
    axs[2].plot(timestamps_ekf, ekf_theta, label='/ekf/pose/orientation/yaw', color='blue', linestyle='-', linewidth=1)
    axs[2].plot(timestamps_real, real_theta, label='/odom/pose/orientation/yaw', color='red', linestyle='-', linewidth=1)
    #axs[2].plot(timestamps_ground_truth, ground_truth_theta, label='/ground_truth/pose/orientation/yaw', color='green', linestyle=':', linewidth=1)
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend(loc='upper right')
    axs[2].set_title('Theta (Orientation) Over Time')
    axs[2].set_ylabel('Theta (rad)')
    axs[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

"""
# Main execution
bag_path = "/home/francesco-masin/bag_files/final"

# New landmarks with IDs and coordinates
landmark_ids = [11, 12, 13, 21, 22, 23, 31, 32, 33]
landmarks_x = [-1.1, -1.1, -1.1, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1]
landmarks_y = [-1.1, 0.0, 1.1, -1.1, 0.0, 1.1, -1.1, 0.0, 1.1]

# Extract data for /odom, /ekf, and /ground_truth
timestamps_real, real_positions = extract_odom_positions_ros2(bag_path, "/odom")
path_positions = extract_path_ros2(bag_path, "/global_path")
#timestamps_ground_truth, ground_truth_positions = extract_odom_positions_ros2(bag_path, "/ground_truth")

# Interpolate /ekf and /ground_truth to match /odom timestamps
if real_positions and ekf_positions:# and ground_truth_positions:
    interpolated_ekf_positions = interpolate_data(timestamps_real, timestamps_ekf, ekf_positions)
    #interpolated_ground_truth_positions = interpolate_data(timestamps_real, timestamps_ground_truth, ground_truth_positions)

    # Compute errors
    errors_ekf = compute_error(real_positions, interpolated_ekf_positions)
    #errors_ground_truth = compute_error(real_positions, interpolated_ground_truth_positions)

    print("Errors between /odom and /ekf:")
    for key, value in errors_ekf.items():
        print(f"{key}: {value}")
    """
    print("\nErrors between /odom and /ground_truth:")
    for key, value in errors_ground_truth.items():
        print(f"{key}: {value}")
    """
    # Plot the trajectories and X, Y, Theta over time
    plot_trajectories(real_positions, interpolated_ekf_positions, landmark_ids, landmarks_x, landmarks_y)
    #plot_x_y_theta(timestamps_real, real_positions, timestamps_real, interpolated_ekf_positions)

