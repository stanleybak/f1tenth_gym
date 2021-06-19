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
    def step_sim(self, cmd):
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

        self.solid_lines = LineCollection([], lw=2, animated=True, color='k', zorder=2)
        ax.add_collection(self.solid_lines)
        self.artist_list.append(self.solid_lines)
        
        self.dotted_lines = LineCollection([], lw=1, animated=True, color='k', ls=':', zorder=1)
        ax.add_collection(self.dotted_lines)
        self.artist_list.append(self.dotted_lines)

        self.rand_pt_marker, = ax.plot([0], [0], 'g--o', lw=1)
        self.artist_list.append(self.rand_pt_marker)

        self.red_xs_data: Tuple[List[float], List[float]] = ([], [])
        self.red_xs, = ax.plot([0], [0], 'rx', ms=6, zorder=4)
        self.artist_list.append(self.red_xs)

        self.black_xs_data: Tuple[List[float], List[float]] = ([], [])
        self.black_xs, = ax.plot([0], [0], 'kx', ms=6, zorder=3)
        self.artist_list.append(self.black_xs)

        self.init_from_node(root)

    def init_from_node(self, node):
        'initialize artists from root'

        solid_paths = self.solid_lines.get_paths()
        dotted_paths = self.dotted_lines.get_paths()
        
        s = node.state
        sx, sy = node.obs[0:2]

        status = s.get_status()

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

                if child_node.children:
                    dotted_paths.append(Path(verts, codes))
                else:
                    solid_paths.append(Path(verts, codes))

                self.init_from_node(child_node)

    def update_rand_pt_marker(self, rand_pt, obs):
        'update random point marker'

        xs = [rand_pt[0], obs[0]]
        ys = [rand_pt[1], obs[1]]

        self.rand_pt_marker.set_data(xs, ys)

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
            
class TreeNode:
    'tree node in search'

    sim_state_class: Optional[SimulationState] = None

    def __init__(self, state: SimulationState, parent=None):
        assert TreeNode.sim_state_class is not None, "TreeNode.sim_state_class should be set first"
        
        self.state: SimulationState = state
        self.obs: np.ndarray = state.get_obs()
        self.parent: Optional[TreeNode] = parent
        
        self.children: Dict[str, TreeNode] = {}

    def count_nodes(self):
        'return the number of nodes countered recursively'

        count = 1

        for c in self.children.values():
            count += c.count_nodes()

        return count

    def expand_children(self, artists):
        'expand all children of this node'

        assert TreeNode.sim_state_class is not None, "TreeNode.sim_state_class should be set first"
        assert not self.children
        assert self.state.get_status() == 'ok'

        solid_paths = artists.solid_lines.get_paths()
        dotted_paths = artists.dotted_lines.get_paths()

        sx, sy = self.obs[0:2]

        for cmd in TreeNode.sim_state_class.get_cmds():
            child_state = deepcopy(self.state)
            child_state.step_sim(cmd)

            child_node = TreeNode(child_state, self)
            self.children[cmd] = child_node

            # update marker
            status = child_state.get_status()

            if status == 'error':
                artists.add_marker('red_x', child_node.obs)
            elif status == 'stop':
                artists.add_marker('black_x', child_node.obs)
            else:
                assert status == 'ok', f"status was {status}"

            # update drawing, add child to dotted lines
            cx, cy = child_node.obs[0:2]

            codes = [Path.MOVETO, Path.LINETO]
            verts = [(cx, cy), (sx, sy)]
            dotted_paths.append(Path(verts, codes))

        # update drawing; add self to solid lines
        if self.parent:
            codes = [Path.MOVETO, Path.LINETO]
            px, py = self.parent.obs[0:2]
            verts = [(px, py), (sx, sy)]
            solid_paths.append(Path(verts, codes))

    def find_closest_leaf(self, obs_pt):
        '''return the leaf closest to the passed in observation point

        returns leaf_node, distance
        '''

        if not self.children:
            if self.state.get_status() != 'ok':
                min_dist = np.inf
                min_node = None
            else:
                min_dist = np.linalg.norm(self.obs - obs_pt)
                min_node = self
        else:
            min_node = None
            min_dist = np.inf
            
            for c in self.children.values():
                node, dist = c.find_closest_leaf(obs_pt)

                if dist < min_dist:
                    min_node = node
                    min_dist = dist

        return min_node, min_dist
    
def init_plot():
    'initialize plotting style'

    matplotlib.use('TkAgg') # set backend

    parent = os.path.dirname(os.path.realpath(__file__))
    p = os.path.join(parent, 'bak_matplotlib.mlpstyle')

    plt.style.use(['bmh', p])

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
        
def run_fuzz_testing(sim_state_class, seed=0):
    'run fuzz testing with the given simulation state class'

    rng = np.random.default_rng(seed=seed)
    init_plot()
    
    obs_data = sim_state_class.get_obs_data()

    assert len(obs_data) >= 2

    fig = plt.figure(figsize=(8, 8))
    xlim = obs_data[0][1:3]
    ylim = obs_data[1][1:3]
    
    ax = plt.axes(xlim=(xlim[0], xlim[1]), ylim=(ylim[0], ylim[1]))
    ax.set_xlabel(obs_data[0][0])
    ax.set_ylabel(obs_data[1][0])

    plt.tight_layout()

    TreeNode.sim_state_class = sim_state_class

    tree_filename = 'root.pkl'
    root = load_root(tree_filename, sim_state_class)

    artists = Artists(ax, root)

    # plot root point
    plt.plot([root.obs[0]], [root.obs[1]], 'ko', ms=5)

    def animate(frame):
        'animate function for funcAnimation'

        if frame > 0:
            print(f"frame: {frame}")

            if frame % 10 == 0:
                save_root(tree_filename, root)
            
            rand_pt = random_point(rng, obs_data)
        
            # find closest point in tree
            node, _ = root.find_closest_leaf(rand_pt)

            if node is None:
                print("Node was None!")
            else:
                artists.update_rand_pt_marker(rand_pt, node.obs)            
                node.expand_children(artists)

        return artists.artist_list

    anim = animation.FuncAnimation(fig, animate, frames=sys.maxsize, interval=1, blit=True)
    plt.show()

    #while root.get_status() == 'ok':
    #    cmd = cmd_list[rng.integers(len(cmd_list))]

    #    root.step_sim(cmd)


    #    paths = lines.get_paths()

    #    for verts in self.cur_sim_lines:
    #        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    #        paths.append(Path(verts, codes))
