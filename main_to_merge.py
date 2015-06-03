import kivy
import ipdb

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.properties import NumericProperty
from kivy.clock import Clock
from kivy.graphics import Rectangle
from random import randint
from generate_stimuli import *
from evaluation import *
from Verbosity import *

from kivy.config import Config
Config.set('graphics', 'resizable', 0)  # don't make the app re-sizeable
#Graphics fix
 #this fixes drawing issues on some phones
Window.clearcolor = (0, 0, 0, 1.)

# ----------- Global objects -------------

__version__ = 'something'

# game spec
spec = {
    "verbose": 1,
    "num_high_scores": 10,
    "highscorefile": "highscores.txt",
    "max_nback": 3,
    "type_stimulus": "animals",
    "present_stimuli": 10,
    "num_stimuli": 5,
    "max_name_length": 10,
    "max_score": 1000000,
    "num_lives": 3,
    "gamename": "NBACK",
    "max_level": 5
}

app = {}
global app

# stimulus
stimulus = Label()
stimulus.x = Window.width/2 - stimulus.width/2
stimulus.y = Window.height/2 - stimulus.height/2

# stimulus array
stimulus_store = []
score = 0

score_display = Label(text="0")
# Position score_display
score_display.x = Window.width*0.5 - score_display.width/2
score_display.y = Window.height*0.9 - ((score_display.width/2)*2)


lives = spec["num_lives"]

# ----------- Functions ------------------





# ----------- Classes --------------------

class WidgetDrawer(Widget):
    #This widget is used to draw all of the objects on the screen
    #it handles the following:
    # widget movement, size, positioning
    #whever a WidgetDrawer object is created, an image string needs to be specified
    #example:    wid - WidgetDrawer('./image.png')
 
    #objects of this class must be initiated with an image string
    #;You can use **kwargs to let your functions take an arbitrary number of keyword arguments
    #kwargs ; keyword arguments
    def __init__(self, imageStr, **kwargs):
        super(WidgetDrawer, self).__init__(**kwargs)  # this is part of the **kwargs notation
        #if you haven't seen with before, here's a link http://effbot.org/zone/python-with-statement.html
        with self.canvas:
            #setup a default size for the object
            self.size = (Window.width*.002*25, Window.width*.002*25)
            #this line creates a rectangle with the image drawn on top
            self.rect_bg = Rectangle(source=imageStr, pos=self.pos, size = self.size)
            #this line calls the update_graphics_pos function every time the position variable is modified
            self.bind(pos=self.update_graphics_pos) 
            self.x = self.center_x
            self.y = self.center_y
            #center the widget 
            self.pos = (self.x,self.y) 
            #center the rectangle on the widget
            self.rect_bg.pos = self.pos 
 
    def update_graphics_pos(self, instance, value):
    #if the widgets position moves, the rectangle that contains the image is also moved
        self.rect_bg.pos = value  
    #use this function to change widget size        
    def setSize(self, width, height): 
        self.size = (width, height)
    #use this function to change widget position    
    def setPos(xpos,ypos):
        self.x = xpos
        self.y = ypos

class MyButton(Button):
    #class used to get uniform button styles
    def __init__(self, **kwargs):
        super(MyButton, self).__init__(**kwargs)
    #all we're doing is setting the font size. more can be done later
        self.font_size = Window.width*0.018
        self.size = Window.width*.3,Window.width*.1
        for key, value in kwargs.iteritems():      # styles is a regular dictionary
            if key == "num":
                self.num = value

        # def press_button(obj):
        # #this function will be called whenever the reset button is pushed
        #     print '%s button pushed' % self.num
        #     # ipdb.set_trace()
        #     # lives = GUI.end_turn(GUI,self.num)

        # self.bind(on_release=press_button) 

class GUI(Widget):
    #this is the main widget that contains the game. 
    def __init__(self, **kwargs):
        super(GUI, self).__init__(**kwargs)
        l = Label(text='NBack') #give the game a title
        l.x = Window.width/2 - l.width/2
        l.y = Window.height*0.8
        self.add_widget(l) #add the label to the screen
        stimulus.text = "Touch to start"
        self.add_widget(stimulus)
        self.started = False
        self.add_widget(score_display)

    stimulus_store = []

    def gameStart(self): 
        oneButton = MyButton(text='One', num=1, pos=(Window.left*0.1,Window.height*0.8))
        twoButton = MyButton(text='Two', num=2, pos=(Window.left*0.1,Window.height*0.6))
        threeButton = MyButton(text='Three', num=3, pos=(Window.left*0.1,Window.height*0.4))

        
        #*** It's important that the parent gets the button so you can click on it
        #otherwise you can't click through the main game's canvas
        self.parent.add_widget(oneButton)
        self.parent.add_widget(twoButton)
        self.parent.add_widget(threeButton)
        # generate the first stimulus
        self.stimulus_store.append(str(generate_stimulus(spec["type_stimulus"],spec["num_stimuli"])))
        print self.stimulus_store
        stimulus.text = str(self.stimulus_store[0])
        def end_turn(self,response):
            global score
            global lives
            # Evaluate the user's response
            answer = evaluate_response(spec['verbose'],self.stimulus_store,response)
            if not answer:
                lives-=1 
                print lives
            # update score based upon evaluation
            else:
                score+=10**response
                # Update displayed score
                score_display.text = str(score)
            if lives<=0:
                self.game_over()
            else:
                # Generate a new stimulus and store it
                new_stim = generate_stimulus(spec["type_stimulus"],spec["num_stimuli"])
                if len(self.stimulus_store) >= spec["max_nback"]:
                    self.stimulus_store.pop(0)
                stimulus_store.append(new_stim)
                verboseprint(spec["verbose"], self.stimulus_store, len(self.stimulus_store))
                stimulus.text = new_stim

        oneButton.bind(on_release=end_turn(self,1))
        twoButton.bind(on_release=end_turn(self,2))
        threeButton.bind(on_release=end_turn(self,3))


    # If necessary terminate the game
    def game_over(self):
        stimulus.text = "GAME OVER"
        self.oneButton.remove_widget()
        self.twoButton.remove_widget()
        self.threeButton.remove_widget()
        self.remove_widget()

    #Every time the screen is touched, the on_touch_down function is called
    def on_touch_down(self, touch):
        if not self.started:
            self.started = True
            # self.end_turn(0)
            self.gameStart()
        # else:
            # self.end_turn(0)

    def update(self,dt):
        #This update function is the main update function for the game
        #All of the game logic has its origin here 
        #events are setup here as well
        # everything here is executed every 60th of a second.
        pass

class ClientApp(App):
    def build(self):
        #this is where the root widget goes
        #should be a canvas
        parent = Widget() #this is an empty holder for buttons, etc
 
        app = GUI()
        #Start the game clock (runs update function once every (1/60) seconds
        Clock.schedule_interval(app.update, 1.0/60.0) 
        parent.add_widget(app) #use this hierarchy to make it easy to deal w/buttons
        return parent





if __name__ == '__main__' :
    ClientApp().run()