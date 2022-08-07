
import System
obj = System.System()

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

root = Tk()
root.geometry("750x400")

def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("image files",  "*.png .jpg .jpeg*"),))

    if(filename !=""):
        obj.manual_prediction(filename)     



button_img1 = PhotoImage(file='manual_prediction.png')
button_img2 = PhotoImage(file='capture_img.png')
button_img3 = PhotoImage(file='live_detection.png')

canvas = Canvas(root, width=750, height=400)

canvas.pack(fill="both", expand=True)
im = ImageTk.PhotoImage(Image.open("wallpaper_dlll.jpg"))
canvas.create_image(0,0, image=im, anchor="nw")

canvas.create_text(350, 50, text="Facial Emotion Recognition",font=("Arial Bold", 25), width=500, fill="white")

b1 = Button(root ,command = browseFiles, activebackground='red', image=button_img1, bg='black')
b2 = Button(root, activebackground='red', image=button_img2, bg='black', command=obj.captureFromCamera)
b3 = Button(root, activebackground='red', image=button_img3, bg='black', command=obj.runMainLoop)

#display buttons
b1_canvas = canvas.create_window(70, 200, anchor="nw", window=b1)
b2_canvas = canvas.create_window(300, 200, anchor="nw", window=b2)
b3_canvas = canvas.create_window(510, 200, anchor="nw", window=b3)

root.mainloop()

obj.release_camera();




