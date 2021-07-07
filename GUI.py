import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from tkinter import *
from tkinter import ttk
import cv2
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing import image
import os
from tkinter import messagebox


#import required libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam

import cv2
import numpy as np
from tensorflow.keras.models import load_model


import time
import tkinter as tk


LARGE_FONT= ("Verdana", 12)


class Project(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="icon.ico")
        tk.Tk.wm_title(self, "Multi Feature Face Recognition")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, PageThree,PageFour):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Facial Emotion Recognition Using Images",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = ttk.Button(self, text="Facial Emotion Recognition Using Web Camera",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()

        button3 = ttk.Button(self, text="Detect Mask",
                            command=lambda: controller.show_frame(PageThree))
        button3.pack()

        button3 = ttk.Button(self, text="Smile and Take a Selfie ",
                            command=lambda: controller.show_frame(PageFour))
        button3.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Facial Emotion Recognition Using Images!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Facial Emotion Recognition Using Web Camera",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()
        self.labelFrame = ttk.LabelFrame(self, text = "Open File")
        self.labelFrame.pack(padx = 20, pady = 20)
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.button()

    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse A File",command = self.fileDialog)
        self.button.grid(column = 1, row = 1)

    def fileDialog(self):

        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("jpeg files","*.jpg"),("all files","*.*")) )
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.label.configure(text = self.filename)

        self.chooseAndPredict(self.filename)

    def chooseAndPredict(self,img):
        self.image = Image.open(img)
        self.python_image = ImageTk.PhotoImage(self.image)
        ttk.Label(self, image=self.python_image).pack()
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Happy", 3: "Sad", 4: "Surprised", 5: "Neutral"}
        
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,1.3,5)
        i = 1
        if not len(faces) :
            messagebox.showinfo("ERROR", "Couldn't detect a face in this image")

        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]  # croping
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                messagebox.showinfo("Emotion of the person", emotion_dict[maxindex])


class PageTwo(tk.Frame):


    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Facial Emotion Recognition Using WebCam!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        
        button3 = ttk.Button(self, text="Facial Emotion Recognition Using Images",
                            command=lambda: controller.show_frame(PageOne))
        button3.pack()


        self.labelFrame = ttk.LabelFrame(self, text = "Open Camera")
        self.labelFrame.pack(padx = 20, pady = 20)

        self.snapshot()

    def snapshot(self) : 
        button2 = ttk.Button(self.labelFrame, text="Take A Picture",
                            command=self.plot_image)
        button2.grid(column = 1, row = 1)
        

    def plot_image(self):
        App(tk.Toplevel(), "Webcam","frame.jpg")
        # self.plot("frame.jpg")
        self.chooseAndPredict()

    def chooseAndPredict(self):
        self.image = Image.open("frame.jpg")
        self.python_image = ImageTk.PhotoImage(self.image)
        ttk.Label(self, image=self.python_image).pack()
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Happy", 3: "Sad", 4: "Surprised", 5: "Neutral"}
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        image = cv2.imread("frame.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,1.3,5)
        i = 1
        if not len(faces) :
                messagebox.showinfo("ERROR", "Couldn't detect a face in this image")
        else :        
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]  # croping
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                messagebox.showinfo("Emotion of the person", emotion_dict[maxindex])

    
    def plot(self,img) :
        self.filename =img
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.label.configure(text = self.filename)


        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        if self.filename :
            img = Image.open(self.filename)
            a.imshow(img)
            canvas = FigureCanvasTkAgg(f, self)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2Tk(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Face Mask Detection !", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self,text="Detect Mask",
                            command=self.detect_mask)
        button2.pack()

    def detect_mask(self):
        model=load_model("model2-001.model")

        labels_dict={0:'without mask',1:'mask'}
        color_dict={0:(0,0,255),1:(0,255,0)}

        size = 4
        webcam = cv2.VideoCapture(0) #Use camera 0

        # We load the xml file
        classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        while True:
            (rval, im) = webcam.read()
            im=cv2.flip(im,1,1) #Flip to act as a mirror

            # Resize the image to speed up detection
            mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

            # detect MultiScale / faces 
            faces = classifier.detectMultiScale(mini)

            # Draw rectangles around each face
            for f in faces:
                (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
                #Save just the rectangle faces in SubRecFaces
                face_img = im[y:y+h, x:x+w]
                resized=cv2.resize(face_img,(150,150))
                normalized=resized/255.0
                reshaped=np.reshape(normalized,(1,150,150,3))
                reshaped = np.vstack([reshaped])
                result=model.predict(reshaped)
                #print(result)
                
                label=np.argmax(result,axis=1)[0]
                
                cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
                cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
                cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                
            # Show the image
            cv2.imshow('Detect Mask',im)
            key = cv2.waitKey(10)
            # if Esc key is press then break out of the loop 
            if key == 27: #The Esc key
                # break
            # Stop video
                webcam.release()

                # Close all started windows
                cv2.destroyAllWindows()


class PageFour(tk.Frame):
    
    def __init__(self, parent, controller):
        self.cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
        self.cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml') 
        self.cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Take Selfie By Detecting Smile!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        
        button3 = ttk.Button(self, text="Facial Emotion Recognition Using Images",
                            command=lambda: controller.show_frame(PageOne))
        button3.pack()


        self.labelFrame = ttk.LabelFrame(self, text = "Open Camera")
        self.labelFrame.pack(padx = 20, pady = 20)

        self.snapshot()
        self.label = ttk.Label(self.labelFrame,text="")
        self.label.grid(column = 1, row = 2)
        # self.picture()

    def snapshot(self) : 
        button2 = ttk.Button(self.labelFrame, text="Take A Selfie Automatically when you smile",
                            command=self.takeselfie)
        button2.grid(column = 1, row = 1)
        
        

    def detection(self,grayscale, img,cnt):
        face = self.cascade_face.detectMultiScale(grayscale, 1.3, 5)
        for (x_face, y_face, w_face, h_face) in face:
            cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 130, 0), 2)
            ri_grayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
            ri_color = img[y_face:y_face+h_face, x_face:x_face+w_face] 
            eye = self.cascade_eye.detectMultiScale(ri_grayscale, 1.2, 18) 
            for (x_eye, y_eye, w_eye, h_eye) in eye:
                cv2.rectangle(ri_color,(x_eye, y_eye),(x_eye+w_eye, y_eye+h_eye), (0, 180, 60), 2) 
            smile = self.cascade_smile.detectMultiScale(ri_grayscale, 1.7, 20)
            for (x_smile, y_smile, w_smile, h_smile) in smile: 
                cv2.rectangle(ri_color,(x_smile, y_smile),(x_smile+w_smile, y_smile+h_smile), (255, 0, 130), 2)
                path=r'C:\Users\user\Desktop\Semester7\Artificial Intelligence\AI Project\img'+time.strftime("%d-%m-%Y-%H-%M-%S")+'.png'
                cv2.imwrite(path,img) 
                cnt+=1
                self.image2 = PhotoImage(file=(path))
                self.label.configure(image=self.image2)
                self.label.image=self.image2
        return img,cnt 


    def takeselfie(self):
        vc = cv2.VideoCapture(0) 
        cnt=0
        while True:
            _, img = vc.read() 
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            final,cnt = self.detection(grayscale, img,cnt) 
            cv2.imshow('Smile to take a selife :)', final)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
            if cnt==1 :
                break
        vc.release() 
        cv2.destroyAllWindows() 


class App:
    def __init__(self, window, window_title,current_picture, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        self.current_picture = current_picture
        # Button that lets the user take a snapshot
        self.btn_snapshot=tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        

        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
 
         # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()
 
    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.current_picture = self.framename()
            cv2.imwrite(self.current_picture, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        self.vid.release()
        self.window.destroy()
        self.window.quit()

    def framename(self): 
        # return "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"
        return "frame.jpg"

    def update(self):
         # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source,cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

     # Release the video source when the object is destroyed
    def release(self):
        if self.vid.isOpened():
            self.vid.release()


def predict_emotion(test_image):
    emotion=['Angry','Disgust','Happy','Sad','Surprise']
    test_image = image.load_img(test_image, target_size = (48,48),color_mode='grayscale')
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    neural_net_output=emotion_model.predict(test_image)[0]
    print(neural_net_output)
    neural_net_output=neural_net_output.tolist()
    print(emotion [neural_net_output.index(max(neural_net_output))])
# create ur CNN Model 
emotion_model = Sequential()

#adding the layers
#step 1 add ur first Conv layers
# (nbr of filter, kernel size (row,col),input shape, activation )
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

#step -2 add the Pooling layer
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

#step -3 add drop out layer
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

#step -4 when you are satified with your feature extractor add Flattening layer.
emotion_model.add(Flatten())

#step -5 add ur Fully connected layers
emotion_model.add(Dense(1024, activation='relu',))
emotion_model.add(Dropout(0.25))
emotion_model.add(Dense(5, activation='softmax'))

emotion_model.load_weights("emotion_model.h5")

app = Project()
app.mainloop()

