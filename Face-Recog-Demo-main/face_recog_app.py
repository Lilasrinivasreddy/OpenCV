
import streamlit as st 
import tempfile
from PIL import Image
import streamlit.components.v1 as stc
import face_recognition
import cv2
import numpy as np


HTML_BANNER = """
<div style="background-color:Orange;padding:10px;border-radius:10px">
<h1 style="color:Black;text-align:center;">Face Recognition</h1>
</div>
"""

DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'test.png'

















stc.html(HTML_BANNER)

st.sidebar.title('Face Recognition')

st.sidebar.text('Params For video')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True)

#IMAGW PART

img_file_buffer = st.sidebar.file_uploader("Upload the Test image", type=[ "jpg", "jpeg",'png'])

imfile = tempfile.NamedTemporaryFile(delete=False)

if  not img_file_buffer:
    image = np.array(Image.open(DEMO_IMAGE))
   
    img_file_buffer= DEMO_IMAGE
        
else:
    

    image = np.array(Image.open(img_file_buffer))
    imfile.write(img_file_buffer.read())

    
name_input = st.sidebar.text_input('Name of the Person',value = 'Rose')
use_webcam = st.sidebar.button('Use Webcam')
video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])

tfflie = tempfile.NamedTemporaryFile(delete=False)

stop_button = st.sidebar.button('Stop Processing')


if stop_button:
    st.stop()



if not video_file_buffer:
    if use_webcam:
        vid = cv2.VideoCapture(0)
            
    else:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tfflie.name = DEMO_VIDEO
    

        
else:
    tfflie.write(video_file_buffer.read())
    vid = cv2.VideoCapture(tfflie.name)

st.sidebar.text('Input Video')
st.sidebar.video(tfflie.name)

#starting with face encoding
person1_img =  face_recognition.load_image_file(img_file_buffer)
given_face_encoding = face_recognition.face_encodings(person1_img)[0]

known_face_encodings = [given_face_encoding]

known_face_name = [name_input]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

stframe = st.empty()

while vid.isOpened():
    ret, frame = vid.read()

    if not ret:
        break
    small_frame = cv2.resize(frame,(0, 0), fx=0.25, fy=0.25)

    new_frame = small_frame[:,:,::-1]

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(new_frame)
        face_encodings = face_recognition.face_encodings(new_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_name[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)



    stframe.image(frame,channels = 'BGR',use_column_width=True)

    









