import time
import threading
from pynput.mouse import Button,Controller

from pynput.keyboard import Listener,KeyCode

delay = 0.01
button = Button.left
start_key= KeyCode(char = 's')
stop_key = KeyCode(char = 'e')
pause_key = KeyCode(char = 'p')

class ClickMouse(threading.Thread):
    def __init__(self,delay,button):
        super(ClickMouse,self).__init__()
        self.delay = delay
        self.button = button
        self.running = False
        self.program_running = True
    def start_click(self):
        self.running = True
    def stop_click(self):
        self.running = False
    def exit_click(self):
        self.stop_click()
        self.program_running = False
    def run(self):
        while self.program_running:
            while self.running:
                mouse.click(self.button)
                time.sleep(self.delay)
            time.sleep(0.1)
mouse = Controller()
Click_thread = ClickMouse(delay,button)
Click_thread.start()
def on_press(key):
    if key == start_key:
        if Click_thread.running:
            Click_thread.stop_click()
        else:
            Click_thread.start_click()
    elif key == pause_key:
            Click_thread.stop_click()
    elif key == stop_key:
        Click_thread.exit_click()
        listener.stop()

with Listener(on_press=on_press) as listener:
    listener.join()

