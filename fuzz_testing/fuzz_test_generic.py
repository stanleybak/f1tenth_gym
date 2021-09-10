'''
Generic fuzz tester for CPS systems
'''

from typing import Dict, List, Tuple, Optional, Callable

import os
import sys
import time
import pickle
from sklearn.cluster import DBSCAN
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.widgets import Button

class SimulationState(ABC):
    'abstract simulation state class'

    @staticmethod
    @abstractmethod
    def get_cmds() -> Tuple[str]:
        """get a tuple of commands (strings) that can be passed into step_sim"""

    @staticmethod
    @abstractmethod
    def get_obs_data():
        """get labels and ranges on observations

        returns:
            list of 3-tuples, label, min, max
        """

    @staticmethod
    @abstractmethod
    def set_nominal():
        """set nominal behavior mode (should have deterministic sim), affects get_cmds()"""

    def __init__(self, is_root=True):
        """initialize root state (if is_root=True) or blank state (otherwise)"""

    @abstractmethod
    def step_sim(self, cmd, debug=False):
        """do one step of the simulation (modifies self)"""

    @abstractmethod
    def get_status(self):
        """get simulation status. element of ['ok', 'stop', 'error']

        'ok' -> continue simulation
        'stop' -> state is not an error, but no longer interesting to continue simuations
        'error' -> start is an error, flag it!
        """

    @abstractmethod
    def render(self):
        """displace a visualization for error traces(optional)"""

    @abstractmethod
    def get_obs(self)->np.ndarray:
        """get an observation of the state

        returns a np.array of float objects
        """

    @abstractmethod
    def get_map_pos(self)->np.ndarray:
        """get the map position the state

        returns a np.array of float objects
        """

    @staticmethod
    def make_map_artist(self, ax):
        """make and return an artist for plotting the map, usingthe passed-in axis (optional)"""

        return None

    @staticmethod
    def select_best_cmd(obs, rand_obs, cmd_options_list) -> Optional[List[str]]:
        """select the cmd from cmd_options_list to use, in order to best expand the state with
        observation 'obs' towards 'rand_pt'

        this allows to use application-specific information in order to optimize an RRT search

        if this not overridden (None is returned), all cmds will be tried
        """

        return None

class Artists:
    'artists for animating tree search'

    def __init__(self, ax, map_ax, root):
        self.artist_list = []

        self.obs_solid_lines = LineCollection([], lw=2, animated=True, color='k', zorder=1)
        ax.add_collection(self.obs_solid_lines)
        self.artist_list.append(self.obs_solid_lines)

        self.map_solid_lines = LineCollection([], lw=2, animated=True, color='k', zorder=1)
        map_ax.add_collection(self.map_solid_lines)
        self.artist_list.append(self.map_solid_lines)
        
        self.rand_pt_marker, = ax.plot([], [], '--o', color='lime', lw=1, zorder=1)
        self.artist_list.append(self.rand_pt_marker)

        self.obs_blue_circle_marker, = ax.plot([], [], 'o', color='b', ms=8, zorder=2)
        self.artist_list.append(self.obs_blue_circle_marker)

        self.obs_green_circle_marker, = ax.plot([], [], 'o', color='g', ms=6, zorder=3)
        self.artist_list.append(self.obs_green_circle_marker)

        self.map_blue_circle_marker, = map_ax.plot([], [], 'o', color='b', ms=8, zorder=2)
        self.artist_list.append(self.map_blue_circle_marker)

        self.obs_red_xs_data: Tuple[List[float], List[float]] = ([], [])
        self.obs_red_xs, = ax.plot([], [], 'rx', ms=6, zorder=4)
        self.artist_list.append(self.obs_red_xs)

        self.map_red_xs_data: Tuple[List[float], List[float]] = ([], [])
        self.map_red_xs, = map_ax.plot([], [], 'rx', ms=6, zorder=4)
        self.artist_list.append(self.map_red_xs)

        self.obs_black_xs_data: Tuple[List[float], List[float]] = ([], [])
        self.obs_black_xs, = ax.plot([], [], 'kx', ms=6, zorder=3)
        self.artist_list.append(self.obs_black_xs)

        self.map_black_xs_data: Tuple[List[float], List[float]] = ([], [])
        self.map_black_xs, = map_ax.plot([], [], 'kx', ms=6, zorder=3)
        self.artist_list.append(self.map_black_xs)

        self.clusters = []

        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, 20)]
        for i in range(20):

            data, = map_ax.plot([], [], 'o', markerfacecolor=tuple(colors[i]),
                                        markeredgecolor='k', markersize=7, zorder=5)
            self.clusters.append(data)
            self.artist_list.append(self.clusters[i])

        assert TreeNode.sim_state_class.make_map_artist is not None
        self.map_artist = TreeNode.sim_state_class.make_map_artist(map_ax)

        if self.map_artist:
            self.artist_list.append(self.map_artist)

        self.init_from_node(root)

    def init_from_node(self, node):
        'initialize artists from root'

        obs_solid_paths = self.obs_solid_lines.get_paths()
        map_solid_paths = self.map_solid_lines.get_paths()
        
        sx, sy = node.obs[0:2]
        smapx, smapy = node.map_pos

        status = node.status

        if status == 'error':
            self.add_marker('red_x', node.obs, node.map_pos)
        elif status in ['stop', 'out_of_bounds']:
            self.add_marker('black_x', node.obs, node.map_pos)
        else:
            #assert status == 'ok', f"status was {status}"

            for child_node in node.children.values():
                cx, cy = child_node.obs[0:2]
                cmapx, cmapy = child_node.map_pos

                codes = [Path.MOVETO, Path.LINETO]
                verts = [(cx, cy), (sx, sy)]
                obs_solid_paths.append(Path(verts, codes))

                codes = [Path.MOVETO, Path.LINETO]
                verts = [(cmapx, cmapy), (smapx, smapy)]
                map_solid_paths.append(Path(verts, codes))

                self.init_from_node(child_node)

    def update_rand_pt_marker(self, rand_pt, obs):
        'update random point marker'

        xs = [rand_pt[0], obs[0]]
        ys = [rand_pt[1], obs[1]]

        self.rand_pt_marker.set_data(xs, ys)

    def update_obs_blue_circle(self, pt):
        'update random point marker'

        if pt is None:
            self.obs_blue_circle_marker.set_data([], [])
        else:
            xs = [pt[0]]
            ys = [pt[1]]

            self.obs_blue_circle_marker.set_data(xs, ys)

    def update_obs_green_circle(self, xs, ys):
        'update in-cache point marker'

        self.obs_green_circle_marker.set_data(xs, ys)

    def update_map_blue_circle(self, pt):
        'update random point marker'

        if pt is None:
            self.map_blue_circle_marker.set_data([], [])
        else:
            xs = [pt[0]]
            ys = [pt[1]]

            self.map_blue_circle_marker.set_data(xs, ys)

    def add_marker(self, type_str, obs, map_pos):
        '''add marker

        type_str: one of 'red_x', 'black_x'
        obs: the observation vector
        '''

        if type_str == 'red_x':
            self.obs_red_xs_data[0].append(obs[0])
            self.obs_red_xs_data[1].append(obs[1])
            self.obs_red_xs.set_data(*self.obs_red_xs_data)

            self.map_red_xs_data[0].append(map_pos[0])
            self.map_red_xs_data[1].append(map_pos[1])
            self.map_red_xs.set_data(*self.map_red_xs_data)
        else:
            assert type_str == 'black_x'
            
            self.obs_black_xs_data[0].append(obs[0])
            self.obs_black_xs_data[1].append(obs[1])
            self.obs_black_xs.set_data(*self.obs_black_xs_data)

            self.map_black_xs_data[0].append(map_pos[0])
            self.map_black_xs_data[1].append(map_pos[1])
            self.map_black_xs.set_data(*self.map_black_xs_data)

def is_out_of_bounds(pt, box):
    'is the pt out of box?'

    rv = False

    for x, (lb, ub) in zip(pt, box):
        if x < lb or x > ub:
            rv = True
            break

    return rv

def find_max_node(node):
    'return index of max node'

    i = node.node_index

    for c in node.children.values():
        i = max(find_max_node(c), i)

    return i

class TreeNode:
    'tree node in search'

    sim_state_class: Optional[SimulationState] = None
    node_counter = 0

    def __init__(self, state: SimulationState, cmd_from_parent=None, parent=None, limits_box=None):
        assert TreeNode.sim_state_class is not None, "TreeNode.sim_state_class should be set first"
        
        self.state: Optional[SimulationState] = state
        self.obs: np.ndarray = state.get_obs()
        self.map_pos: np.ndarray = state.get_map_pos()
        
        self.status: str = state.get_status()
        self.limits_box = limits_box
        self.cmd_from_parent = cmd_from_parent

        if limits_box is not None and is_out_of_bounds(self.obs, limits_box):
            self.status = 'out_of_bounds'
        
        self.parent: Optional[TreeNode] = parent
        
        self.children: Dict[str, TreeNode] = {}
        self.node_index = TreeNode.node_counter
        TreeNode.node_counter += 1

    def get_open_cmds(self) -> List[str]:
        """Get list of unexplored commands from this node"""

        cmd_list = list(TreeNode.sim_state_class.get_cmds())

        for cmd in self.children.keys():
            cmd_list.remove(cmd)

        return cmd_list

    def get_cache_key(self) -> str:
        """get the cache key string for this node"""

        cmd_list = self.get_cmd_list()
        all_cmds = TreeNode.sim_state_class.get_cmds()

        cmd_indices = [str(all_cmds.index(cmd)) for cmd in cmd_list]

        # using blank string for join is fine as long as num commands < 10
        join_str = "" if len(all_cmds) < 10 else "."
        
        cache_key = join_str.join(cmd_indices)
        
        return cache_key

    def get_cmd_list(self) -> List[str]:
        """get commands leading to this node from the root, in order"""

        rv = []

        if self.parent:
            rv = self.parent.get_cmd_list()
            rv.append(self.cmd_from_parent)

        return rv

    def count_nodes(self):
        """return the number of nodes countered recursively"""

        count = 1

        for c in self.children.values():
            count += c.count_nodes()

        return count

    def reconstruct_state(self, tree_search_obj):
        """reconstruct self.state"""

        assert self.state is None

        cur_node = self
        node_list = []

        while True:
            node_list.insert(0, cur_node)

            # root should always have state
            assert cur_node.parent is not None
            cur_node = cur_node.parent

            if cur_node.state is not None:
                break

        # replay cmd_list starting from node, inserting into cache if needed
        cur_state = deepcopy(cur_node.state)

        for node in node_list:
            cmd = node.cmd_from_parent

            cur_state.step_sim(cmd)

            cur_obs = cur_state.get_obs()

            assert np.allclose(cur_obs, node.obs)
            assert cur_state.get_status() == 'ok'

            # avoid a copy for self
            node.state = cur_state if node is self else deepcopy(cur_state)

            tree_search_obj.add_to_cache(node)

        assert self.state is not None

    def expand_child(self, tree_search_obj, cmd, allow_repeat_children=False):
        """expand the given child of this node"""

        artists = tree_search_obj.artists
        obs_limits_box = tree_search_obj.obs_limits_box

        assert TreeNode.sim_state_class is not None, "TreeNode.sim_state_class should be set first"

        if not allow_repeat_children:
            assert cmd not in self.children
                            
        assert self.status == 'ok'

        obs_solid_paths = artists.obs_solid_lines.get_paths()
        map_solid_paths = artists.map_solid_lines.get_paths()

        sx, sy = self.obs[0:2]
        smapx, smapy = self.map_pos

        # reconstruct self.state if it doesn't exist
        if self.state is None:
            self.reconstruct_state(tree_search_obj)

        assert self.state is not None

        child_state = deepcopy(self.state)

        child_state.step_sim(cmd)

        child_node = TreeNode(child_state, cmd_from_parent=cmd, parent=self, limits_box=obs_limits_box)

        if cmd in self.children:
            # cmd is already in children
            old_child = self.children[cmd] 
            old_child.node_index = child_node.node_index # update node index

            child_node = old_child
        else:
            self.children[cmd] = child_node

            # update marker
            status = child_node.status

            if status == 'error':
                artists.add_marker('red_x', child_node.obs, child_node.map_pos)
            elif status in ['stop', 'out_of_bounds']:
                artists.add_marker('black_x', child_node.obs, child_node.map_pos)
            else:
                assert status == 'ok', f"status was {status}"

            # update drawing, add child to solid lines
            cx, cy = child_node.obs[0:2]
            cmapx, cmapy = child_node.map_pos

            codes = [Path.MOVETO, Path.LINETO]
            verts = [(cx, cy), (sx, sy)]
            obs_solid_paths.append(Path(verts, codes))

            codes = [Path.MOVETO, Path.LINETO]
            verts = [(cmapx, cmapy), (smapx, smapy)]
            map_solid_paths.append(Path(verts, codes))

        if child_node.status == 'ok':
            tree_search_obj.add_to_cache(child_node)
        else:
            child_node.state = None

    def dist(self, p, q):
        'distance between two points'

        xscale = 1
        yscale = 1

        if self.limits_box:
            xscale = self.limits_box[0][1] - self.limits_box[0][0]
            yscale = self.limits_box[1][1] - self.limits_box[1][0]

        dx = (p[0] - q[0]) / xscale
        dy = (p[1] - q[1]) / yscale

        return np.linalg.norm([dx, dy])

    def find_closest_map_node(self, map_pt):
        """recursively find the node closest to the passed in map point

        returns node, distance
        """
        
        min_node = None
        min_dist = np.inf

        if self.status == "error":
            min_dist = np.linalg.norm(self.map_pos - map_pt)
            min_node = self
            
        for c in self.children.values():
            node, dist = c.find_closest_map_node(map_pt)

            if dist < min_dist:
                min_node = node
                min_dist = dist

        return min_node, min_dist

    def find_closest_node(self, obs_pt, filter_func: Callable[..., bool]):
        """recursively find the node closest to the passed in observation point

        filter_func selects if a node should be considered, for example you can use
            open_node_filter_func or click_filter_func

        returns node, distance
        """
        
        min_node = None
        min_dist = np.inf

        if filter_func(self):
            min_dist = self.dist(self.obs, obs_pt)
            min_node = self
            
        for c in self.children.values():
            node, dist = c.find_closest_node(obs_pt, filter_func)

            if dist < min_dist:
                min_node = node
                min_dist = dist

        return min_node, min_dist

def random_point(rng, obs_data):
    'generate random point in range for rrt'

    randvec = rng.random(len(obs_data))
    rv = []

    for r, odata in zip(randvec, obs_data):
        lb, ub = odata[1:3]
        
        x = lb + r * (ub - lb)
        rv.append(x)

    return np.array(rv, dtype=float)

class TreeSearch:
    'performs and draws the tree search'

    def __init__(self, seed, nominal, always_from_start, sim_state, max_nodes, cache_size):
        self.always_from_start = always_from_start
        self.cur_node = None # used if always_from_start = True

        self.rng = np.random.default_rng(seed=seed)
        self.last_save_count = 0

        rrt = 'rrt' if not always_from_start else 'rand'
        
        if nominal:
            rrt = 'nominal'
            always_from_start = False
            TreeNode.sim_state_class.set_nominal()
        
        classname = sim_state.get_pickle_name()

        self.tree_filename = f'cache/root_{classname}_{rrt}_{seed}.pkl'
        self.root = None

        # key is ','.join(cmd_list)
        # values are guaranteed to have sim_state != None
        self.node_cache: Dict[str, TreeNode] = {}
        self.cache_size = cache_size
        
        self.artists = None

        self.obs_data = TreeNode.sim_state_class.get_obs_data()
        self.obs_limits_box = tuple((lb, ub) for _, lb, ub in self.obs_data)

        self.paused = False
        self.max_nodes = max_nodes

        self.fig = None
        self.ax = None
        self.map_ax = None
        self.init_plot()

    def init_plot(self):
        'initalize plotting'

        obs_data = self.obs_data
        assert len(obs_data) >= 2, "need at least two coordinates to plot"

        matplotlib.use('TkAgg') # set backend

        parent = os.path.dirname(os.path.realpath(__file__))
        p = os.path.join(parent, 'bak_matplotlib.mlpstyle')

        plt.style.use(['bmh', p])

        self.fig, ax_list = plt.subplots(1, 2, figsize=(14, 8))
        
        xlim = obs_data[0][1:3]
        ylim = obs_data[1][1:3]

        self.ax = ax_list[0]
        self.map_ax = ax_list[1]
        
        #self.ax = plt.axes(xlim=(xlim[0], xlim[1]), ylim=(ylim[0], ylim[1]))
        self.ax.set_xlim(xlim[0], xlim[1])
        self.ax.set_ylim(ylim[0], ylim[1])

        self.map_ax.set_xlim(-80, 80)
        self.map_ax.set_ylim(-80, 80)
        
        self.ax.set_xlabel(obs_data[0][0])
        self.ax.set_ylabel(obs_data[1][0])

        self.map_ax.set_xlabel("Map X")
        self.map_ax.set_ylabel("Map Y")

        plt.subplots_adjust(bottom=0.2)
        
        self.bstart = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Start/Stop')
        self.bstart.on_clicked(self.button_start_stop)

        self.bclusters = Button(plt.axes([0.5, 0.05, 0.1, 0.075]), 'Find Clusters')
        self.bclusters.on_clicked((self.button_clusters))


        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.mouse_click)

        #bprev = Button(axprev, 'Reset')
        #bprev.on_clicked(self.button_reset)

        #plt.tight_layout()

    def animate_to_node(self, node):
        'animate to node'

        cmds: List[str] = []
        count = 10

        while node.parent is not None and count > 0:
            cmds = [node.cmd_from_parent] + cmds
            node = node.parent
            count -= 1

        if node.state is None:
            node.reconstruct_state(self)
            
        state = deepcopy(node.state)
        
        for cmd in cmds:
            state.step_sim(cmd)

    def mouse_click(self, event):
        'mouse click event callback'

        if self.paused and event.xdata is not None:
            x, y = event.xdata, event.ydata
            pt = np.array([x, y], dtype=float)
            node = None

            if event.inaxes == self.ax:
                node, _ = self.root.find_closest_node(pt, click_filter_func)
            elif event.inaxes == self.map_ax:
                node, _ = self.root.find_closest_map_node(pt)

            if node is not None:
                self.animate_to_node(node)

    def mouse_move(self, event):
        'mouse move event callback'

        self.artists.update_obs_blue_circle(None)
        self.artists.update_map_blue_circle(None)

        if self.paused and event.xdata is not None:
            x, y = event.xdata, event.ydata
            pt = np.array([x, y], dtype=float)
            node = None

            if event.inaxes == self.ax:
                node, _ = self.root.find_closest_node(pt, click_filter_func)
            elif event.inaxes == self.map_ax:
                # mouse is on map axis
                node, _ = self.root.find_closest_map_node(pt)

            if node is not None:
                self.artists.update_obs_blue_circle(node.obs)
                self.artists.update_map_blue_circle(node.map_pos)

    def button_start_stop(self, _event):
        'start/stop button pressed callback'

        self.paused = not self.paused
        print(f"Paused: {self.paused}")

        if not self.paused:
            self.artists.update_obs_blue_circle(None)
            self.artists.update_map_blue_circle(None)
            for i in range(20):
                self.artists.clusters[i].set_data([], [])

    def button_clusters(self, _event):
        'start/stop button pressed callback'

        if not self.paused:
            return
        crash_nodes = self.find_crashes(self.root)
        map_obs = self.find_map_obs(crash_nodes)
        clusters = DBSCAN(eps=3, min_samples=3).fit(map_obs)
        labels = clusters.labels_
        unique_labels = set(clusters.labels_)

        for k in unique_labels:
            class_member_mask = (labels == k)
            xy = np.array(map_obs)[np.array(class_member_mask)]
            self.artists.clusters[k].set_data(xy[:,0], xy[:,1])
        print(map_obs)
        print (clusters.labels_)

    def find_crashes(self, root):
        crash_lst = []
        if root.status == "error":
            crash_lst.append(root)
        for _,child_node in root.children.items():
            crash_lst+=self.find_crashes(child_node)
        return crash_lst

    def find_map_obs(self, crash_nodes):
        map_obs = []
        for node in crash_nodes:
            map_obs.append(node.map_pos)
        return map_obs

    def animate(self, frame):
        'animate function for funcAnimation'

        assert TreeNode.sim_state_class is not None
        assert self.artists is not None
        assert self.root is not None

        if frame > 0 and not self.paused:
            # sometimes save nodes
            count = TreeNode.node_counter

            if not self.always_from_start:
                assert count == self.root.count_nodes()

            if count >= 2 * self.last_save_count:
                self.save_root()

            if count >= self.max_nodes:
                self.save_root()
                self.paused = True
                print(f"Paused. Reached max_nodes: {self.max_nodes}")
            else:
                self.compute_next_point()

        return self.artists.artist_list

    def remove_from_cache(self, node):
        """remove a node from the cache (and set state to None)"""

        node_key = node.get_cache_key()

        del self.node_cache[node_key]

    def add_to_cache(self, node):
        """add a treenode to the cache"""

        assert node.state.get_status() == 'ok'

        if len(self.node_cache) >= self.cache_size:
            key_list = list(self.node_cache.keys())

            # evict one at random
            index = self.rng.integers(len(key_list))
            remove_key = key_list[index]
            assert remove_key != ''  # root shouldn't be in cache

            print(f"removing {remove_key} from node cache")

            remove_node = self.node_cache[remove_key]
            remove_node.state = None  # clear

            del self.node_cache[remove_key]

            # update cached states in plot
            xs = []
            ys = []
            for n in self.node_cache.values():
                assert n.state is not None

                xs.append(n.obs[0])
                ys.append(n.obs[1])

            assert self.artists is not None
            self.artists.update_obs_green_circle(xs, ys)

        node_key = node.get_cache_key()
        self.node_cache[node_key] = node

    def save_root(self):
        """save root node"""

        start = time.perf_counter()
        print("saving... ", end='')

        key_list = list(self.node_cache.keys())
        cached_states = [self.node_cache[k].state for k in key_list]

        # clear states before saving
        for k in key_list:
            self.node_cache[k].state = None

        raw = pickle.dumps(self.root)
        mb = len(raw) / 1024 / 1024

        with open(self.tree_filename, "wb") as f:
            f.write(raw)

        diff = time.perf_counter() - start
        count = TreeNode.node_counter #self.root.count_nodes()
        kb_per = 1024 * mb / count
        self.last_save_count = count

        # retore states after saving
        for index, k in enumerate(key_list):
            self.node_cache[k].state = cached_states[index]

        print(f"saved {count} nodes ({round(mb, 2)} MB, {round(kb_per, 1)} KB per state) in " +
              f"{round(1000 * diff, 1)} ms to {self.tree_filename}")

    def compute_next_point_rrt(self):
        """compute next point using rrt"""

        # RRT-like strategy
        rand_pt = random_point(self.rng, self.obs_data)

        # find closest point in tree
        node, _ = self.root.find_closest_node(rand_pt, open_node_filter_func)

        if node is None:
            print("Closest node was None! Full tree was expanded. Paused.")
            self.paused = True
        else:
            self.artists.update_rand_pt_marker(rand_pt, node.obs)

            all_cmds = node.get_open_cmds()

            # if there's only one choice, don't nccd to filter
            if len(all_cmds) == 1:
                expand_cmd = None
            else:
                expand_cmd = TreeNode.sim_state_class.select_best_cmd(node.obs, rand_pt, all_cmds)

            if expand_cmd is not None:
                node.expand_child(self, expand_cmd)
            else:
                for expand_cmd in all_cmds:
                    node.expand_child(self, expand_cmd)

    def compute_next_point_from_start(self):
        """compute next point using always-from-start strategy"""

        # always from start strategy
        status = self.cur_node.status

        if status != 'ok':
            print(f"resetting to root due to cur_node.status: {status}")
            self.cur_node = self.root

        use_saved_state_optimization = False
            
        while True:
            cmd_list = TreeNode.sim_state_class.get_cmds()
            cmd = cmd_list[self.rng.integers(len(cmd_list))]

            if not use_saved_state_optimization:
                break # replay the state, possibly multiple times
            
            if cmd in self.cur_node.children:
                self.cur_node = self.cur_node.children[cmd]
            elif self.cur_node.status != 'ok':
                self.cur_node = self.root
            else:
                break

        # expand use_last_node using cmd
        allow_repeat_children = not use_saved_state_optimization
        self.cur_node.expand_child(self, cmd, allow_repeat_children=allow_repeat_children)
        self.cur_node = self.cur_node.children[cmd]

        self.artists.update_rand_pt_marker(self.cur_node.obs, self.cur_node.obs)

    def compute_next_point(self):
        """compute next point (in animiaton loop)"""

        if self.cur_node is None:
            assert not self.always_from_start
            self.compute_next_point_rrt()
        else:
            assert self.always_from_start
            self.compute_next_point_from_start()

    def load_root(self, root_sim_state, load_progress_from_file):
        'load root node from pickled file'

        # important for initializing renderer
        self.root = TreeNode(root_sim_state)
        count = 1
        max_node = 0

        if load_progress_from_file:
            try:
                with open(self.tree_filename, "rb") as f:
                    self.root = pickle.load(f)
                    assert self.root.state is not None
                    count = self.root.count_nodes()
                    max_node = find_max_node(self.root)
            except FileNotFoundError:
                pass

        if count == 1:
            print("initialized new search tree (1 node)")
        else:
            print(f"initialized tree with {count} nodes, max_node={max_node}")

        TreeNode.node_counter = max_node + 1
        self.last_save_count = count

    def run(self, root_sim_state, load_progress_from_file):
        'run the search'

        assert self.ax is not None and self.map_ax is not None

        self.load_root(root_sim_state, load_progress_from_file)

        if self.always_from_start:
            self.cur_node = self.root

        self.artists = Artists(self.ax, self.map_ax, self.root)

        # plot root point (not animated)
        self.ax.plot([self.root.obs[0]], [self.root.obs[1]], 'ko', ms=5)

        _anim = animation.FuncAnimation(self.fig, self.animate, frames=sys.maxsize, interval=1, blit=True)
        plt.show()

def open_node_filter_func(tree_node: TreeNode) -> bool:
    """filter function for find_closest_node, which chooses nodes with unexplored commands"""

    rv = False

    if tree_node.status == "ok" and tree_node.get_open_cmds():
        rv = True

    return rv

def click_filter_func(tree_node: TreeNode) -> bool:
    """filter function for find_closest_node, which chooses nodes that can be clicked on"""

    # currently this is not ok nodes or leaves
    
    rv = False

    if tree_node.status == "error" or not tree_node.children:
        rv = True

    return rv

def run_fuzz_testing(sim_state, seed=0, nominal=False, always_from_start=False, max_nodes=1023, cache_size=sys.maxsize,
                     load_progress_from_file=False):
    'run fuzz testing with the given simulation state class'

    TreeNode.sim_state_class = type(sim_state)

    search = TreeSearch(seed, nominal, always_from_start, sim_state, max_nodes, cache_size=cache_size)

    search.run(sim_state, load_progress_from_file)
