"""
code to do the mapping gui for implicit hybrid surfaces
"""

import os
import sys

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.widgets import Button

from PIL import Image
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

class Artists:
    """Artists for gui"""

    def __init__(self, map_ax, gym_sim):
        self.artists_list = []
        self.gym_sim = gym_sim

        self.single_path = [] # single controller path, lists of x,y pairs

        self.map_solid_lines = LineCollection([], lw=2, animated=True, color='k', zorder=1)
        map_ax.add_collection(self.map_solid_lines)
        self.artist_list.append(self.map_solid_lines)
        
        self.map_black_dots, = map_ax.plot([], [], 'o', color='l', ms=6, zorder=1)
        self.artist_list.append(self.map_black_dots)

        self.map_artist = self.make_map_artist(map_ax, gym_sim.map_conig_dict)
        self.artist_list.append(self.map_artist)

    def add_to_single_path(self, x, y):
        """add point to single controller path"""

        WORKING_HERE()

    def make_map_artist(self, ax, map_config_dict):
        """make and return an artist for plotting the map, usingthe passed-in axis (optional)"""

        # map static map artist
        image_path = map_config_dict['image']
        res = map_config_dict['resolution']
        origin = map_config_dict['origin']

        img = Image.open(image_path)
        img = img.convert("RGBA")

        pixdata = img.load()

        width, height = img.size
        for y in range(height):
            for x in range(width):
                if pixdata[x, y] == (255, 255, 255, 255):
                    pixdata[x, y] = (255, 255, 255, 0)

        img = np.array(img)

        xsize, ysize = img.shape[0:2]
        
        x1 = origin[0]
        y1 = origin[1]
        assert origin[2] == 0

        xsize *= res
        ysize *= res

        box = Bbox.from_bounds(x1, y1, xsize, ysize)
        tbox = TransformedBbox(box, ax.transData)
        box_image = BboxImage(tbox, zorder=2)

        box_image.set_data(img)
        box_image.zorder = 0

        ax.add_artist(box_image)

        ax.set_xlim(x1, x1 + xsize)
        ax.set_ylim(y1, y1 + ysize)
        
        return box_image

class MapGui:
    """container object for map gui"""

    def __init__(self, gym_sim):
        self.map_ax = None
        self.fig = None
        self.artists = None

        self.gym_sim = gym_sim

        self.paused = False

    def show_gui(self):
        """show matplotlib GUI"""

        matplotlib.use('TkAgg') # set backend

        parent = os.path.dirname(os.path.realpath(__file__))
        p = os.path.join(parent, 'bak_matplotlib.mlpstyle')

        plt.style.use(['bmh', p])

        self.fig, self.map_ax = plt.subplots(1, 1, figsize=(10, 8))

        # gets set again later
        self.map_ax.set_xlim(-80, 80)
        self.map_ax.set_ylim(-80, 80)

        self.map_ax.set_xlabel("Map X")
        self.map_ax.set_ylabel("Map Y")

        plt.subplots_adjust(bottom=0.2)

        self.fig.canvas.mpl_connect('button_press_event', self.mouse_click)

        self.artists = Artists(self.map_ax, self.gym_sim)

        _anim = animation.FuncAnimation(self.fig, self.animate, frames=sys.maxsize, interval=1, blit=True)
        plt.show()

    def mouse_click(self, event):
        'mouse click event callback'

        if self.paused and event.xdata is not None:
            x, y = event.xdata, event.ydata
            pt = np.array([x, y], dtype=float)

    def animate(self, frame):
        'animate function for funcAnimation'

        assert self.artists is not None

        if frame > 0 and not self.paused:
            if self.gym_sim.finished_initial_lap:
                x, y = self.gym_sim.step_initial_lap()

                # add x y to artists
                self.artists.add_to_single_path(x, y)

            else:
                print("Finished initial lap, pause")
                self.paused = True

        return self.artists.artist_list

        
