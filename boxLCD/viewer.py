import numpy as np
import pyglet

class Viewer:
  """use pyglet to render images that have already been generated, to show to user live"""
  def __init__(self, width, height, C):
    self.window = pyglet.window.Window(2*width + 2, height)
    self.width = width+2
    self.height = height
    self.C = C
    self.lcd_w = int(self.C.lcd_base*self.C.wh_ratio)
    self.lcd_h = self.C.lcd_base

  def render(self, image, return_rgb_array=False):
    """
    image (np.ndarray): shape (H,W,C) in RGB 0-255 uint8 format
    return_rgb_array (bool): if you want the pyglet buffer back
    """
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    img = pyglet.image.ImageData(image.shape[1], image.shape[0], 'RGB', image.tobytes(), pitch=image.shape[1] * -3)
    img.blit(0, 0)
    if self.C.debug:
      label1 = pyglet.text.HTMLLabel(f'<font face="Helvetica Bold" size=10">{self.height}x{self.width}x3</font>', x=self.width/2, y=7*self.height/8, anchor_x='center', anchor_y='center')
      label1.draw()
      label2 = pyglet.text.HTMLLabel(f'<font face="Helvetica Bold" size=10">{self.lcd_h}x{self.lcd_w}</font>', x=3*self.width/2, y=7*self.height/8, anchor_x='center', anchor_y='center')
      label2.draw()
    arr = None
    if return_rgb_array:
      buffer = pyglet.image.get_buffer_manager().get_color_buffer()
      image_data = buffer.get_image_data()
      arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
      arr = arr.reshape(buffer.height, buffer.width, 4)
      arr = arr[::-1, :, 0:3]
    self.window.flip()
    return arr