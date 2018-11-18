from tkinter import Tk, Label, Button
import os
import cv2

from tkinter import *
from tkinter.ttk import Progressbar
from tkinter import ttk
from tkinter import Menu
from tkinter import messagebox
from tkinter import font  as tkfont
from PIL import ImageTk, Image

import tensorflow as tf

root = Tk()
C = Canvas(root, bg="black", height=650, width=500)
filename = PhotoImage(file="/Users/JackMa/Desktop/MLProject/Logo.png")
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


class FreeFormGUI:
    def __init__(self, master):
        self.master = master
        master.title("Free Form")

        self.title_font = tkfont.Font(family='Comic Sans', size=18, weight="bold", slant="italic")
        self.label = Label(master, text="Welcome to Free Form | There is no form to communication", font=self.
                           title_font, fg='white', bg='black')
        self.label.pack(side=TOP, padx=5, pady=14)

        button_font = tkfont.Font(family='Comic Sans', size=14, weight="bold")
        self.greet_button = Button(master, text="Begin your imagination", command=self.free_form, height=5, width=25)
        self.greet_button['font'] = button_font
        self.greet_button.pack(side=BOTTOM, pady=15)

    def free_form(self):
        os.system("python /Users/JackMa/Desktop/MLProject/freeform.py")


my_gui = FreeFormGUI(root)

menu = Menu(root)
menu.add_command(label='File')
root.config(menu=menu)
new_item = Menu(menu)
new_item.add_command(label='FAQ')
new_item.add_command(label='Credits')
menu.add_cascade(label='Options', menu=new_item)
root.config(menu=menu)

C.pack()
root.geometry('650x470')
root.mainloop()



