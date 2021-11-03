"""
Pyglet testing
"""

import time
import math

import pyglet
from pyglet import gl

class MyRenderer(pyglet.window.Window):
    """overrided renderer"""

    def __init__(self):
        width = 600
        height = 600
        conf = gl.Config(sample_buffers=1,
                      samples=4,
                      depth_size=16,
                      double_buffer=True)
        super().__init__(width, height, config=conf, resizable=True, vsync=False)

        # gl init
        gl.glClearColor(9/255, 32/255, 87/255, 1.)

        self.batch = pyglet.graphics.Batch()

        self.score_label = pyglet.text.Label(
                'Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}'.format(
                    laptime=0.0, count=0.0),
                font_size=36,
                x=0,
                y=-400,
                anchor_x='center',
                anchor_y='center',
                # width=0.01,
                # height=0.01,
                color=(255, 255, 255, 255),
                batch=self.batch)

        self.fps_display = pyglet.window.FPSDisplay(self)

        self.left = -500
        self.right = 500
        self.top = 500
        self.bottom = -500

        self.closed = False

    def on_draw(self, extra_draw_func=None):
        """
        Function when the pyglet is drawing. The function draws the batch created that includes the map points, the agent polygons, and the information text, and the fps display.
        
        Args:
            None

        Returns:
            None
        """

        # Initialize Projection matrix
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        # Initialize Modelview matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        # Save the default modelview matrix
        gl.glPushMatrix()

        # Clear window with ClearColor
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Set orthographic projection matrix
        gl.glOrtho(self.left, self.right, self.bottom, self.top, 1, -1)

        # Draw all batches
        self.batch.draw()
        self.fps_display.draw()

        if extra_draw_func is not None:
            extra_draw_func()
        
        # Remove default modelview matrix
        gl.glPopMatrix()

    def on_close(self):
        """
        Callback function when the 'x' is clicked on the window, overrides inherited method. Also throws exception to end the python program when in a loop.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: with a message that indicates the rendering window was closed
        """

        super().on_close()
        self.closed = True

def main():
    """main entry point"""

    w = MyRenderer()

    pt_list = (0, 0, 0, 100, 100, 0, 200, 0, 0, 300, 100, 0)

    # extra drawing
    num_pts = 4
    green = (0, 255, 0)
    red = (255, 0, 0)
    vertex_list = w.batch.add(4, pyglet.gl.GL_LINES, None,
            ('v3f/stream', pt_list),
            ('c3B/static', green + red + green + red)
        )

    #label = pyglet.text.Label('Hello World!', font_name='Arial',font_size=36, x=0, y=0)
    
    counter = 0

    #label_list = [None]

    #def extra_draw_func():
    #    if label_list[0]:
    #        label_list[0].draw()

    old_label = None

    # render loop
    while not w.closed:
        w.dispatch_events()
        w.on_draw()
        w.flip()

        counter += 1
        theta = counter / 50

        x = 200 * math.cos(theta)
        y = 200 * math.sin(theta)

        vertex_list.vertices[3:5] = 2*x, y

        if old_label is not None:
            old_label.delete()
        
        old_label = pyglet.text.Label('Hello World',
        font_size=20,
        x=2*x,
        y=y,
        anchor_x='center',
        anchor_y='center',
        color=(255, 255, 255, 255), batch=w.batch)

        #label_list[0] = label

        #print(x, y)

        #time.sleep(0.01)

if __name__ == "__main__":
    main()
