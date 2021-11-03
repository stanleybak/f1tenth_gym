#!/usr/bin/env python

import pyglet

window = pyglet.window.Window()
label = pyglet.text.Label('Hello World!',font_name='Arial',font_size=36, x=0, y=0)

@window.event                       
def on_mouse_motion(x, y, dx, dy):
    window.clear()
    label.x = x
    label.y = y


@window.event
def on_draw():
    label.draw()

pyglet.app.run() 
