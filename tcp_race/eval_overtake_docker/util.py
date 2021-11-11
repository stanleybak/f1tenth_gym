"""
race utilities
"""

def pack_odom(obs, i):
    """create single-car odometry from multi-car odometry"""

    keys = {
        'poses_x': 'pose_x',
        'poses_y': 'pose_y',
        'poses_theta': 'pose_theta',
        'linear_vels_x': 'linear_vel_x',
        'linear_vels_y': 'linear_vel_y',
        'ang_vels_z': 'angular_vel_z',
    }
    return {single: obs[multi][i] for multi, single in keys.items()}

def get_pose(obs, i):
    """extra pose from observation"""

    x = obs['poses_x'][i]
    y = obs['poses_y'][i]
    theta = obs['poses_theta'][i]

    return x, y, theta
