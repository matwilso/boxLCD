import pyglet
set_width = False
# this is a really bad idea if the underlying code changes.
# i really should make my own rendering class to copy the gym.envs.classic_control.rendering code.

class Viewer:
  def __init__(self, width, height, C):
    self.window = pyglet.window.Window(2*width, height)
    self.width = width
    self.height = height
    self.C = C
    self.lcd_w = int(self.C.lcd_base*self.C.wh_ratio)
    self.lcd_h = self.C.lcd_base

  def render(self, image, return_rgb_array=False):
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    img = pyglet.image.ImageData(image.shape[1], image.shape[0], 'RGB', image.tobytes(), pitch=image.shape[1] * -3)
    img.blit(0, 0)
    label1 = pyglet.text.HTMLLabel(f'<font face="Times New Roman" size=10">({self.width}x{self.height}x3)</font>', x=self.width/2, y=3*self.height/4, anchor_x='center', anchor_y='center')
    label1.draw()
    label2 = pyglet.text.HTMLLabel(f'<font face="Times New Roman" size=10">({self.lcd_w}x{self.lcd_h})</font>', x=3*self.width/2, y=3*self.height/4, anchor_x='center', anchor_y='center')
    label2.draw()
    arr = None
    if return_rgb_array:
      buffer = pyglet.image.get_buffer_manager().get_color_buffer()
      image_data = buffer.get_image_data()
      arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
      # In https://github.com/openai/gym-http-api/issues/2, we
      # discovered that someone using Xmonad on Arch was having
      # a window of size 598 x 398, though a 600 x 400 window
      # was requested. (Guess Xmonad was preserving a pixel for
      # the boundary.) So we use the buffer height/width rather
      # than the requested one.
      arr = arr.reshape(buffer.height, buffer.width, 4)
      arr = arr[::-1, :, 0:3]
    self.window.flip()
    return arr