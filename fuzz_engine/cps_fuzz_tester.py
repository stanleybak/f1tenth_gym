'''
Generic fuzz tester for CPS systems
'''

from typing import Dict, List, Tuple, Optional

import os
import sys
import time
import pickle

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
    def get_cmds()->List[str]:
        'get a list of commands (strings) that can be passed into step_sim'

    @staticmethod
    @abstractmethod
    def get_obs_data():
        '''get labels and ranges on observations

        returns:
            list of 3-tuples, label, min, max
        '''

    def __init__(self, is_root=True):
        'initialize root state (if is_root=True) or blank state (otherwise)'

    @abstractmethod
    def step_sim(self, cmd, debug=False):
        'do one step of the simulation (modifies self)'

    @abstractmethod
    def get_status(self):
        '''get simulation status. element of ['ok', 'stop', 'error']

        'ok' -> continue simulation
        'stop' -> state is not an error, but no longer interesting to continue simuations
        'error' -> start is an error, flag it!
        '''

    @abstractmethod
    def render(self):
        'displace a visualization for error traces(optional)'

    @abstractmethod
    def get_obs(self)->np.ndarray:
        '''get an observation of the state

        returns a list of float objects
        '''

class Artists:
    'artists for animating tree search'

    def __init__(self, ax, root):
        self.artist_list = []

        self.solid_lines = LineCollection([], lw=2, animated=True, color='k', zorder=1)
        ax.add_collection(self.solid_lines)
        self.artist_list.append(self.solid_lines)
        
        self.rand_pt_marker, = ax.plot([], [], '--o', color='lime', lw=1, zorder=1)
        self.artist_list.append(self.rand_pt_marker)

        self.blue_circle_marker, = ax.plot([], [], 'o', color='b', ms=8, zorder=2)
        self.artist_list.append(self.blue_circle_marker)

        self.red_xs_data: Tuple[List[float], List[float]] = ([], [])
        self.red_xs, = ax.plot([], [], 'rx', ms=6, zorder=4)
        self.artist_list.append(self.red_xs)

        self.black_xs_data: Tuple[List[float], List[float]] = ([], [])
        self.black_xs, = ax.plot([], [], 'kx', ms=6, zorder=3)
        self.artist_list.append(self.black_xs)

        self.init_from_node(root)

    def init_from_node(self, node):
        'initialize artists from root'

        solid_paths = self.solid_lines.get_paths()
        
        sx, sy = node.obs[0:2]

        status = node.status

        if status == 'error':
            self.add_marker('red_x', node.obs)
        elif status == 'stop':
            self.add_marker('black_x', node.obs)
        else:
            assert status == 'ok', f"status was {status}"

            for child_node in node.children.values():
                cx, cy = child_node.obs[0:2]

                codes = [Path.MOVETO, Path.LINETO]
                verts = [(cx, cy), (sx, sy)]
                solid_paths.append(Path(verts, codes))

                self.init_from_node(child_node)

    def update_rand_pt_marker(self, rand_pt, obs):
        'update random point marker'

        xs = [rand_pt[0], obs[0]]
        ys = [rand_pt[1], obs[1]]

        self.rand_pt_marker.set_data(xs, ys)

    def update_blue_circle(self, pt):
        'update random point marker'

        if pt is None:
            self.blue_circle_marker.set_data([], [])
        else:
            xs = [pt[0]]
            ys = [pt[1]]

            self.blue_circle_marker.set_data(xs, ys)

    def add_marker(self, type_str, obs):
        '''add marker

        type_str: one of 'red_x', 'black_x'
        obs: the observation vector
        '''

        if type_str == 'red_x':
            self.red_xs_data[0].append(obs[0])
            self.red_xs_data[1].append(obs[1])
            self.red_xs.set_data(*self.red_xs_data)
        else:
            assert type_str == 'black_x'
            
            self.black_xs_data[0].append(obs[0])
            self.black_xs_data[1].append(obs[1])
            self.black_xs.set_data(*self.black_xs_data)

def is_out_of_bounds(pt, box):
    'is the pt out of box?'

    rv = False

    for x, (lb, ub) in zip(pt, box):
        if x < lb or x > ub:
            rv = True
            break

    return rv
            
class TreeNode:
    'tree node in search'

    sim_state_class: Optional[SimulationState] = None

    def __init__(self, state: SimulationState, cmd_from_parent=None, parent=None, limits_box=None):
        assert TreeNode.sim_state_class is not None, "TreeNode.sim_state_class should be set first"
        
        self.state: SimulationState = state
        self.obs: np.ndarray = state.get_obs()
        self.status: str = state.get_status()
        self.limits_box = limits_box
        self.cmd_from_parent = cmd_from_parent

        if limits_box is not None and is_out_of_bounds(self.obs, limits_box):
            self.status = 'out_of_bounds'
        
        self.parent: Optional[TreeNode] = parent
        
        self.children: Dict[str, TreeNode] = {}

    def get_cmd_list(self):
        'get commands leading to this node'

        rv = []

        if self.parent:
            rv = self.parent.get_cmd_list()
            rv.append(self.cmd_from_parent)

        return rv

    def count_nodes(self):
        'return the number of nodes countered recursively'

        count = 1

        for c in self.children.values():
            count += c.count_nodes()

        return count

    def expand_child(self, artists, cmd, obs_limits_box):
        'expand the given child of this node'

        assert TreeNode.sim_state_class is not None, "TreeNode.sim_state_class should be set first"
        assert not cmd in self.children
        assert self.status == 'ok'

        solid_paths = artists.solid_lines.get_paths()

        sx, sy = self.obs[0:2]

        print(f"\nexpand_child() called with cmd: {cmd}")
        print(f"stored next command: {self.state.next_cmds}")

        node = self

        while node is not None:
            print(f"parent obs state: {node.obs}")
            node = node.parent

        child_state = deepcopy(self.state)

        child_state.step_sim(cmd, debug=True)
        print(f"after step_sim(), child observed state: {child_state.get_obs()}")

        child_node = TreeNode(child_state, cmd_from_parent=cmd, parent=self, limits_box=obs_limits_box)
        self.children[cmd] = child_node

        ########################33
        print("DEBUG: Checking path from root...")
        cmds = child_node.get_cmd_list()
        print(f"cmds: {cmds}")

        # find root
        root = self

        while root.parent is not None:
            root = root.parent
        
        state = deepcopy(root.state)
        
        for i, c in enumerate(cmds):
            print(f"replay obs state: {state.get_obs()}")

            #debug = i == len(cmds) - 1
            debug = False

            if debug:
                print(f"in replay, stored next command: {state.next_cmds}")
            
            state.step_sim(c, debug=debug)

        resim_obs = state.get_obs()

        print(f"re-simulate observed state: {resim_obs}")

        if not np.allclose(child_node.state.get_obs(), resim_obs):
            print("Error: Path from root mismatch!")
            exit(1)

        print("Path from root matches.")

        print(f"next cmds from replay: {state.next_cmds}")
        print(f"next cmds from child: {child_state.next_cmds}")

        if not np.allclose(state.next_cmds, child_state.next_cmds):
            print("!!!next command mismatch!!!\n\n")
            exit(1)
        
        ##############################3

        # update marker
        status = child_node.status

        if status == 'error':
            artists.add_marker('red_x', child_node.obs)
        elif status in ['stop', 'out_of_bounds']:
            artists.add_marker('black_x', child_node.obs)
        else:
            assert status == 'ok', f"status was {status}"

        # update drawing, add child to solid lines
        cx, cy = child_node.obs[0:2]

        codes = [Path.MOVETO, Path.LINETO]
        verts = [(cx, cy), (sx, sy)]
        solid_paths.append(Path(verts, codes))

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

    def find_closest_leaf(self, obs_pt, only_ok=True):
        '''return the node closest to the passed in observation point

        returns leaf_node, distance
        '''
        
        min_node = None
        min_dist = np.inf

        if not self.children:
            if self.status == 'ok' or not only_ok:
                min_dist = self.dist(self.obs, obs_pt)
                min_node = self
            
        for c in self.children.values():
            node, dist = c.find_closest_leaf(obs_pt, only_ok=only_ok)

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

def load_root(filename, sim_state_class):
    'load root node from pickled file'

    # important for initializing renderer
    root = TreeNode(sim_state_class())

    try:
        with open(filename, "rb") as f:
            root = pickle.load(f)
    except FileNotFoundError:
        pass

    print(f"initialized tree with {root.count_nodes()} nodes")

    return root

def save_root(filename, root):
    'save search tree to pickled file'

    start = time.perf_counter()

    raw = pickle.dumps(root)
    mb = len(raw) / 1024 / 1024

    with open(filename, "wb") as f:
        f.write(raw)
        
    diff = time.perf_counter() - start
    count = root.count_nodes()
    kb_per = 1024 * mb / count
    
    print(f"saved {count} nodes ({round(mb, 2)} MB, {round(kb_per, 1)} KB per state) in " + \
          f"{round(1000 * diff, 1)} ms to {filename}")

class TreeSearch:
    'performs and draws the tree search'

    def __init__(self, seed=0, always_from_start=False):
        self.always_from_start = always_from_start
        self.cur_node = None # used if always_from_start = True

        self.rng = np.random.default_rng(seed=seed)
        self.tree_filename = 'root.pkl'
        self.root = None
        self.artists = None

        self.obs_data = TreeNode.sim_state_class.get_obs_data()
        self.obs_limits_box = tuple((lb, ub) for _, lb, ub in self.obs_data)

        self.paused = False
        
        self.fig = None
        self.ax = None
        self.init_plot()

    def init_plot(self):
        'initalize plotting'

        obs_data = self.obs_data
        assert len(obs_data) >= 2, "need at least two coordinates to plot"

        matplotlib.use('TkAgg') # set backend

        parent = os.path.dirname(os.path.realpath(__file__))
        p = os.path.join(parent, 'bak_matplotlib.mlpstyle')

        plt.style.use(['bmh', p])

        self.fig = plt.figure(figsize=(8, 8))
        xlim = obs_data[0][1:3]
        ylim = obs_data[1][1:3]

        self.ax = plt.axes(xlim=(xlim[0], xlim[1]), ylim=(ylim[0], ylim[1]))
        self.ax.set_xlabel(obs_data[0][0])
        self.ax.set_ylabel(obs_data[1][0])

        plt.subplots_adjust(bottom=0.2)
        
        self.bstart = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Start/Stop')
        self.bstart.on_clicked(self.button_start_stop)

        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.mouse_click)

        #bprev = Button(axprev, 'Reset')
        #bprev.on_clicked(self.button_reset)

        #plt.tight_layout()

    def animate_to_node(self, node):
        'animate to node'

        cmds = node.get_cmd_list()

        state = deepcopy(self.root.state)
        
        for cmd in cmds:
            print(cmd)
            state.step_sim(cmd)

        print("done!")

    def mouse_click(self, event):
        'mouse click event callback'

        if self.paused and event.inaxes == self.ax and event.xdata is not None:
            x, y = event.xdata, event.ydata

            pt = np.array([x, y], dtype=float)
            node, _ = self.root.find_closest_leaf(pt, only_ok=False)

            self.animate_to_node(node)

    def mouse_move(self, event):
        'mouse move event callback'

        if self.paused and event.inaxes == self.ax and event.xdata is not None:
            x, y = event.xdata, event.ydata

            pt = np.array([x, y], dtype=float)
            node, _ = self.root.find_closest_leaf(pt, only_ok=False)

            self.artists.update_blue_circle(node.obs)
        else:
            self.artists.update_blue_circle(None)

    def button_start_stop(self, _event):
        'start/stop button pressed callback'

        self.paused = not self.paused
        print(f"Paused: {self.paused}")

        if not self.paused:
            self.artists.update_blue_circle(None)

    def animate(self, frame):
        'animate function for funcAnimation'

        if frame > 0 and not self.paused:
            if frame % 10 == 0:
                save_root(self.tree_filename, self.root)

            if self.cur_node is None:
                # RRT-like strategy
                rand_pt = random_point(self.rng, self.obs_data)

                # find closest point in tree
                node, _ = self.root.find_closest_leaf(rand_pt, only_ok=True)

                if node is None:
                    print("Node was None!")
                else:
                    self.artists.update_rand_pt_marker(rand_pt, node.obs)

                    # expand all children
                    for cmd in TreeNode.sim_state_class.get_cmds():
                        node.expand_child(self.artists, cmd, self.obs_limits_box)
            else:
                # always from start strategy
                status = self.cur_node.status

                if status != 'ok':
                    print(f"resetting to root due to cur_node.status: {status}")
                    self.cur_node = self.root

                while True:
                    cmd_list = TreeNode.sim_state_class.get_cmds()
                    cmd = cmd_list[self.rng.integers(len(cmd_list))]

                    if cmd in self.cur_node.children:
                        self.cur_node = self.cur_node.children[cmd]
                    elif self.cur_node.status != 'ok':
                        self.cur_node = self.root
                    else:
                        break

                # expand use_last_node using cmd
                self.cur_node.expand_child(self.artists, cmd, self.obs_limits_box)
                self.cur_node = self.cur_node.children[cmd]

                self.artists.update_rand_pt_marker(self.cur_node.obs, self.cur_node.obs)

        return self.artists.artist_list

    def run(self):
        'run the search'

        self.root = load_root(self.tree_filename, TreeNode.sim_state_class)

        if self.always_from_start:
            self.cur_node = self.root

        self.artists = Artists(self.ax, self.root)

        # plot root point (not animated)
        self.ax.plot([self.root.obs[0]], [self.root.obs[1]], 'ko', ms=5)

        anim = animation.FuncAnimation(self.fig, self.animate, frames=sys.maxsize, interval=1, blit=True)
        plt.show()

def run_fuzz_testing(sim_state_class, seed=0, always_from_start=False):
    'run fuzz testing with the given simulation state class'

    TreeNode.sim_state_class = sim_state_class
    search = TreeSearch(seed, always_from_start)

    search.run()
