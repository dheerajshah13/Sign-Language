import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
model = load_model('signs.h5')



classes = { 0:' This is 0',
            1:'This is 1',
            2:'This is 2',
            3:'This is 3',
            4:'This is 4',
            5:'This is 5',
            6:'This is 6',
            7:'This is 7',
            8:'This is 8',
            9:'This is 9'
}

top=tk.Tk()
top.geometry('800x600')
top.title(' Digit sign classification')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    #image = tf.image.decode_jpeg(image, channels )
    #image = tf.cast(image, tf.float32)
    image = image.resize((224,224))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image= tf.image.convert_image_dtype(image, dtype=tf.float32, saturate=False)

    print(image.shape)
    #image = tf.image.convert_image_dtype(image, dtype=tf.int32, saturate=False)
    pred = model.predict([image])[0]
    pred_1= np.argmax(pred)
    #sign = classes[pred+1]
    #print(sign)
    sign = classes[pred_1]
    print(sign)
    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Know Your Digit from hand sign",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
