import numpy as np
from scipy.spatial.transform import Rotation as R

def apply_rotation_via_matrix(quaternion, theta_x):
    """
    Apply a rotation around the x-axis to a given quaternion using matrix multiplication.

    Args:
        quaternion (np.array): The original quaternion as [x, y, z, w].
        theta_x (float): The angle by which to rotate around the x-axis, in radians.

    Returns:
        np.array: The new quaternion after applying the rotation.
    """
    # Convert the input quaternion to a Rotation object
    original_rotation = R.from_quat(quaternion)

    # Convert the quaternion to a rotation matrix (3x3)
    original_matrix = original_rotation.as_matrix()

    # Create a rotation matrix for the x-axis rotation by theta_x
    rotation_x = R.from_euler('x', theta_x).as_matrix()

    # Apply the rotation by multiplying the matrices: result_matrix = rotation_x * original_matrix
    result_matrix = np.dot(rotation_x, original_matrix)

    # Convert the resulting rotation matrix back to a quaternion
    new_rotation = R.from_matrix(result_matrix)

    # Return the resulting quaternion
    return new_rotation.as_quat()

# Example usage
original_quaternion = np.array([0.702265, -0.698547, 0.0933354, -0.100721])  # Identity quaternion (no rotation)
theta_x = np.radians(0.3)  # Rotate 90 degrees around the x-axis

new_quaternion = apply_rotation_via_matrix(original_quaternion, theta_x)
print("New quaternion after rotation around x-axis using matrix multiplication:", new_quaternion)
