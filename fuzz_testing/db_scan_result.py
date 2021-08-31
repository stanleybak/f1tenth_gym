"""
Stanley Bak
DB Scan Result of test generation
"""

import sys
import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from sklearn.cluster import DBSCAN

from fuzz_test_gym import F110GymSim
from fuzz_test_generic import TreeNode
from fuzz_test_smooth_blocking import *

def display_gui(root):
    """display gui given the root node"""

    matplotlib.use('TkAgg') # set backend

    parent = os.path.dirname(os.path.realpath(__file__))
    p = os.path.join(parent, 'bak_matplotlib.mlpstyle')

    plt.style.use(['bmh', p])

    obs_data = TreeNode.sim_state_class.get_obs_data()

    fig, ax_list = plt.subplots(1, 2, figsize=(10, 6))
    ax, map_ax = ax_list

    xlim = obs_data[0][1:3]
    ylim = obs_data[1][1:3]
        
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    map_ax.set_xlim(-80, 80)
    map_ax.set_ylim(-80, 80)

    ax.set_xlabel(obs_data[0][0])
    ax.set_ylabel(obs_data[1][0])

    map_ax.set_xlabel("Map X")
    map_ax.set_ylabel("Map Y")

    map_config_dict = {'image': 'Spielberg_map.png', 'resolution': 0.05796, 
                       'origin': [-84.85359914210505, -36.30299725862132, 0.000000]}
        
    map_artist = root.state.make_map_artist(map_ax, map_config_dict)

    plt.subplots_adjust(bottom=0.3)

    collisions_obs, collisions_map = get_collisions(root)
    collisions_map_array = np.array(collisions_map, dtype=float)
    collisions_obs_array = np.array(collisions_obs, dtype=float)

    ax.plot(*zip(*collisions_obs), 'rx', ms=3, zorder=1)
    map_ax.plot(*zip(*collisions_map), 'rx', ms=3, zorder=1)

    # clusters
    map_clusters = []
    obs_clusters = []

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, 30)]

    # prepend some colors
    colors = ['lime', 'orange', 'cyan', 'yellow', 'green', 'red', 'pink', 'magenta', 'skyblue',
              'greenyellow', 'peachpuff'] + colors

    num_clusters = len(colors)
    
    for i in range(num_clusters):
        c = colors[i]

        if not isinstance(c, str):
            c = tuple(colors[i])
        
        data_map, = map_ax.plot([], [], 'o', markerfacecolor=c,
                            markeredgecolor='k', markersize=7, zorder=2)
        data_obs, = ax.plot([], [], 'o', markerfacecolor=c,
                            markeredgecolor='k', markersize=7, zorder=2)
                
        map_clusters.append(data_map)
        obs_clusters.append(data_obs)

    # sliders
    axcolor = 'white'
    pos_list = [
                plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor),
                plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)]

    sliders = []
    sliders.append(Slider(pos_list[0], 'eps', 0.1, 15.0, valinit=3.0))
    sliders.append(Slider(pos_list[1], 'min_samples', 1, 10, valinit=3))

    rax = plt.axes([0.025, 0.07, 0.10, 0.10], facecolor=axcolor)
    radio = RadioButtons(rax, ('map-space', 'obs-space'), active=0)

    def update(_):
        'update plot based on sliders'

        r = radio.value_selected
        on_map = False

        if r == 'map-space':
            on_map = True

        eps = sliders[0].val
        min_samples = int(round(sliders[1].val))

        collisions = collisions_map if on_map else collisions_obs

        db_clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(collisions)

        labels = db_clusters.labels_
        unique_labels = set(db_clusters.labels_)

        print(f"eps: {eps}, min_samples: {min_samples}, on_map: {on_map}, num_clusters: {len(unique_labels)}")
        
        if len(unique_labels) > num_clusters:
            print(f"Warning: num clusters ({len(unique_labels)}) exceeds max ({num_clusters})")

        for map_cluster in map_clusters:
            map_cluster.set_data([], [])

        for obs_cluster in obs_clusters:
            obs_cluster.set_data([], [])

        for k in unique_labels:
            if k < 0 or k >= num_clusters:
                continue
            
            class_member_mask = (labels == k)
            
            xy_map = collisions_map_array[np.array(class_member_mask)]
            map_clusters[k].set_data(xy_map[:, 0], xy_map[:, 1])

            xy_obs = collisions_obs_array[np.array(class_member_mask)]
            obs_clusters[k].set_data(xy_obs[:, 0], xy_obs[:, 1])

        # update
        fig.canvas.draw_idle()

    # listeners
    radio.on_clicked(update)
            
    for s in sliders:
        s.on_changed(update)

    # update once (runs db-scan)
    update(None)

    plt.show()

def get_collisions(node):
    """get collision points recursively, for both axes
    returns a pair: collision_points_obs, collision_points_map

    """

    collision_points_obs = []
    collision_points_map = []

    if node.status == 'error':
        collision_points_obs.append(node.obs)
        collision_points_map.append(node.map_pos)

    for child_node in node.children.values():
        o, m = get_collisions(child_node)

        collision_points_obs += o
        collision_points_map += m

    return collision_points_obs, collision_points_map

def main():
    """main entry point"""

    assert len(sys.argv) == 2, "expected single argument: [cache_filename]"

    TreeNode.sim_state_class = F110GymSim

    tree_filename = sys.argv[1]
    root = None

    try:
        with open(tree_filename, "rb") as f:
            root = pickle.load(f)
            assert root.state is not None
            count = root.count_nodes()
            print(f"Loaded {count} nodes")
    except FileNotFoundError as e:
        print(e)

    assert root is not None, "Loading tree from {tree_filename} failed"

    display_gui(root)

if __name__ == "__main__":
    main()
