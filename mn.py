import cv2
import numpy as np
import pygame
import mediapipe as mp
import time

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Pygame for sound playback
pygame.mixer.init()

# Load sounds for different notes
note_files = ['A0.wav', 'B0.wav', 'Bb0.wav', 'C1.wav', 'Db1.wav', 'D1.wav', 'Eb1.wav', 'E1.wav', 'F1.wav', 'Gb1.wav', 'G1.wav', 'Ab1.wav', 'A1.wav', 'Bb1.wav', 'B1.wav', 'C2.wav', 'Db2.wav', 'D2.wav', 'Eb2.wav', 'E2.wav', 'F2.wav', 'Gb2.wav', 'G2.wav', 'Ab2.wav', 'A2.wav', 'Bb2.wav', 'B2.wav', 'C3.wav', 'Db3.wav', 'D3.wav', 'Eb3.wav', 'E3.wav', 'F3.wav', 'Gb3.wav', 'G3.wav', 'Ab3.wav', 'A3.wav', 'Bb3.wav', 'B3.wav', 'C4.wav', 'Db4.wav', 'D4.wav', 'Eb4.wav', 'E4.wav', 'F4.wav', 'Gb4.wav', 'G4.wav', 'Ab4.wav', 'A4.wav', 'Bb4.wav', 'B4.wav', 'C5.wav', 'Db5.wav', 'D5.wav', 'Eb5.wav', 'E5.wav', 'F5.wav', 'Gb5.wav', 'G5.wav', 'Ab5.wav', 'A5.wav', 'Bb5.wav', 'B5.wav', 'C6.wav', 'Db6.wav', 'D6.wav', 'Eb6.wav', 'E6.wav', 'F6.wav', 'Gb6.wav', 'G6.wav', 'Ab6.wav', 'A6.wav', 'Bb6.wav', 'B6.wav', 'C7.wav', 'Db7.wav', 'D7.wav', 'Eb7.wav', 'E7.wav', 'F7.wav', 'Gb7.wav', 'G7.wav', 'Ab7.wav', 'A7.wav', 'Bb7.wav', 'B7.wav', 'C8.wav']  # Replace with your sound files
sounds = [pygame.mixer.Sound('hand2music\\notes\\{}'.format(note)) for note in note_files]

# Function to map hand position to sound and pitch modulation
def play_note_and_modulate_pitch(x, y, frame_width, frame_height):
    # Divide the screen horizontally into sections for different notes
    num_sections = len(sounds)
    section_width = frame_width // num_sections
    
    # Find the current section based on x position (horizontal)
    section_index = min(x // section_width, num_sections - 1)
    
    # Get the sound for the current section
    sound = sounds[section_index]
    
    # Modulate pitch based on the y position (vertical)
    # The closer to the top (y=0), the higher the pitch; the closer to the bottom, the lower the pitch
    pitch_factor = 1 + (y - frame_height / 2) / (frame_height / 2)  # Scale factor for pitch modulation
    pitch_factor = max(0.5, min(2.0, pitch_factor))  # Clamp pitch to avoid too extreme values

    # Adjust playback speed/pitch (Pygame doesn't natively support pitch, but we can use other libraries for this)
    # Here we simply control volume as an example. For real pitch modulation, use libraries like pydub/librosa.
    sound.set_volume(pitch_factor / 2)  # Example of using pitch_factor to adjust volume

    # Play the sound
#    if not pygame.mixer.get_busy():  # Play the sound only if nothing is playing to avoid overlaps
    sound.play()

# Start video capture
cap = cv2.VideoCapture(0)

cTime=0
pTime=0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get coordinates of index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            x = int(index_finger_tip.x * frame.shape[1])  # x-coordinate (horizontal)
            y = int(index_finger_tip.y * frame.shape[0])  # y-coordinate (vertical)
            
            # Draw the hand landmarks on the frame for visualization
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Play note and modulate pitch based on hand position
            play_note_and_modulate_pitch(x, y, frame.shape[1], frame.shape[0])

    # Show the camera feed
    fframe=cv2.flip(frame,1)
    cv2.imshow('Hand Tracking', fframe) 

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(fframe ,str(int(fps)),(10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0),3)


    if cv2.waitKey(5) & 0xFF == 27:  # Exit if the 'Esc' key is pressed
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
