import cv2
import numpy as np
import pygame
import mediapipe as mp
import pysine
import multiprocessing as mpr

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)
sounds = [note for note in range(30,4000)]

pygame.mixer.init()
# Function to map hand position to sound and pitch modulation
def play_note_and_modulate_pitch(x, y, frame_width, frame_height,sounds):
    num_sections = len(sounds)
    section_width = (frame_width*10) // num_sections
    
    section_index = min(x // section_width, num_sections - 1)
    
    sound = sounds[section_index]
    
    pitch_factor = 1 + (y - frame_height / 2) / (frame_height / 2)
    pitch_factor = max(0.5, min(2.0, pitch_factor))  
    pysine.sine(sound)


class camera_read:
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
                index_finger_tip = hand_landmarks.landmark[8]
                x = int(index_finger_tip.x * frame.shape[1])  # x-coordinate (horizontal)
                y = int(index_finger_tip.y * frame.shape[0])  # y-coordinate (vertical)
                
                # Draw the hand landmarks on the frame for visualization
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Play note and modulate pitch based on hand position
                play_note_and_modulate_pitch(x, y, frame.shape[1], frame.shape[0],sounds)

        # Show the camera feed
        fframe=cv2.flip(frame,1)
        cv2.imshow('Hand Tracking', fframe)
        
        if cv2.waitKey(5) & 0xFF == 27:  # Exit if the 'Esc' key is pressed
            break
    cap.release()
    cv2.destroyAllWindows()




#    if not pygame.mixer.get_busy():  # Play the sound only if nothing is playing to avoid overlaps
#        sound.play()

p1=mpr.Process(target=camera_read)
p2=mpr.Process(target=play_note_and_modulate_pitch)

p1.start()
p2.start()



pygame.mixer.quit()
