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
from fuzz_test_gap_follower import *

def display_gui(root):
    """display gui given the root node"""

    save_tree_pdf = False
    save_map_pdf = True

    matplotlib.use('TkAgg') # set backend

    parent = os.path.dirname(os.path.realpath(__file__))
    p = os.path.join(parent, 'bak_matplotlib.mlpstyle')

    plt.style.use(['bmh', p])

    obs_data = TreeNode.sim_state_class.get_obs_data()

    if save_tree_pdf:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        ax_list = [ax, None]
    elif save_map_pdf:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax_list = [None, ax]
    else:
        fig, ax_list = plt.subplots(1, 2, figsize=(10, 6))
        
    ax, map_ax = ax_list

    collisions_obs, collisions_map = get_collisions(root)
    collisions_map_array = np.array(collisions_map, dtype=float)
    collisions_obs_array = np.array(collisions_obs, dtype=float)

    if ax is not None:

        xlim = obs_data[0][1:3]
        ylim = obs_data[1][1:3]

        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])

        ax.set_xlabel(obs_data[0][0])
        ax.set_ylabel(obs_data[1][0])

        ax.plot(*zip(*collisions_obs), 'rx', ms=9, zorder=1)
        ax.plot([root.obs[0] + 0.5], [root.obs[1]], 'o', color='lime', ms=10, zorder=2)

    if map_ax is not None:
        map_ax.set_xlim(-80, 80)
        map_ax.set_ylim(-80, 80)

        map_ax.set_xlabel("Map X Position")
        map_ax.set_ylabel("Map Y Position")

        map_config_dict = {'image': 'Spielberg_map.png', 'resolution': 0.05796, 
                           'origin': [-84.85359914210505, -36.30299725862132, 0.000000]}

        map_artist = root.state.make_map_artist(map_ax, map_config_dict)

        map_ax.plot(*zip(*collisions_map), 'rx', ms=9, zorder=1)

        map_ax.plot([root.map_pos[0] + 10], [root.map_pos[1]], marker=r'$\leftarrow \mathrm{start}$',
                    color='k', ms=60, zorder=2)

    draw_paths(root, ax, map_ax)

    if not save_tree_pdf:
        if not save_map_pdf:
            plt.subplots_adjust(bottom=0.3)

        # clusters
        map_clusters = []
        obs_clusters = []

        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, 30)]

        # prepend some colors
        colors = ['lime', 'orange', 'cyan', 'yellow', 'pink', 'magenta', 'green', 'skyblue',
                  'greenyellow', 'peachpuff', 'lime', 'orange', 'cyan', 'yellow', 'pink', 'magenta', 'skyblue'] + colors

        num_clusters = len(colors)

        for i in range(num_clusters):
            c = colors[i]

            if not isinstance(c, str):
                c = tuple(colors[i])

            if map_ax is not None:
                data_map, = map_ax.plot([], [], 'o', markerfacecolor=c,
                                        markeredgecolor='k', markersize=11, zorder=2)
                map_clusters.append(data_map)

            if ax is not None:
                data_obs, = ax.plot([], [], 'o', markerfacecolor=c,
                                    markeredgecolor='k', markersize=11, zorder=2)

                obs_clusters.append(data_obs)

        def get_db_scan_params():
            """get the db-scan params selected by the gui

            returns radio.value_selected, eps, min_samples
            """

            rv = "map-space", 2.1, 3

            if not save_map_pdf and not save_tree_pdf:
                rv = radio.value_selected, sliders[0].val, int(round(sliders[1].val))

            return rv
                
        def update(_):
            """update plot based on sliders"""

            r, eps, min_samples = get_db_scan_params()
            on_map = False

            if r == 'map-space':
                on_map = True

            collisions = collisions_map if on_map else collisions_obs

            db_clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(collisions)

            labels = db_clusters.labels_
            unique_labels = set(db_clusters.labels_)

            class_member_mask = (labels == -1)
            xy_map = collisions_map_array[np.array(class_member_mask)]
            num_outliers = xy_map.shape[0]
            num_clusters = len(unique_labels) - 1 # don't include outlier class

            print(f"eps: {eps}, min_samples: {min_samples}, on_map: {on_map}, " +
                  f"num_clusters: {num_clusters}, num_outliers: {num_outliers}")

            if len(unique_labels) > num_clusters + 1:
                print(f"Warning: num clusters ({len(unique_labels)}) exceeds max ({num_clusters})")

            for map_cluster in map_clusters:
                map_cluster.set_data([], [])

            for obs_cluster in obs_clusters:
                obs_cluster.set_data([], [])

            for k in unique_labels:
                if k < 0 or k >= num_clusters:
                    continue

                class_member_mask = (labels == k)

                if map_clusters:
                    xy_map = collisions_map_array[np.array(class_member_mask)]
                    map_clusters[k].set_data(xy_map[:, 0], xy_map[:, 1])

                if obs_clusters:
                    xy_obs = collisions_obs_array[np.array(class_member_mask)]
                    obs_clusters[k].set_data(xy_obs[:, 0], xy_obs[:, 1])

            # update canvas
            fig.canvas.draw_idle()
                
        # sliders
        if not save_map_pdf:
            axcolor = 'white'
            pos_list = [
                plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor),
                plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)]

            sliders = []
            sliders.append(Slider(pos_list[0], 'eps', 0.1, 15.0, valinit=2.1))
            sliders.append(Slider(pos_list[1], 'min_samples', 1, 10, valinit=3))

            rax = plt.axes([0.025, 0.07, 0.10, 0.10], facecolor=axcolor)
            radio = RadioButtons(rax, ('map-space', 'obs-space'), active=0)

            # listeners
            radio.on_clicked(update)

            for s in sliders:
                s.on_changed(update)

        # update once (runs db-scan)
        update(None)

    # show canvas or save pdf
        
    if save_map_pdf or save_tree_pdf:
        if len(sys.argv) > 2:
            t = sys.argv[2]
            
            if ax:
                ax.set_title(t)
                
            if map_ax:
                map_ax.set_title(t)
                
        plt.tight_layout()

        if save_map_pdf:
            filename = 'map.png'
        else:
            filename = 'objective_space.pdf'
        
        plt.savefig(filename)
        print(f"saved {filename}")
    else:
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

def draw_paths(node, obs_ax, map_ax):
    """draw paths on axes"""

    for child in node.children.values():
        if obs_ax is not None:
            xs = [node.obs[0], child.obs[0]]
            ys = [node.obs[1], child.obs[1]]
            obs_ax.plot(xs, ys, 'k-', lw=1.0, zorder=0)

        if map_ax is not None:
            xs = [node.map_pos[0], child.map_pos[0]]
            ys = [node.map_pos[1], child.map_pos[1]]
            map_ax.plot(xs, ys, 'k-', lw=1.0, zorder=0)

        draw_paths(child, obs_ax, map_ax)

def main():
    """main entry point"""

    assert len(sys.argv) >= 2, "expected at least single argument: [cache_filename] (plot title)"

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
