import cv2
import numpy as np
import pygame
import mediapipe as mp
import multiprocessing as mpr

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pygame.mixer.init()

# Initialize Pygame for sound playback
def note_frequency(octave_range):
    frequency=[]
    for octave in octave_range:
        for pitch in range(12):
            frequency.append(round(440*2**((octave-4)+(pitch-9)/12),2))
    return frequency
frequency=note_frequency(range(4,6))

# Function to map hand position to sound and pitch modulation
def play_note_and_modulate_pitch(x, y, frame_width, frame_height,frequency):
    freq_to_play=frequency[x//(frame_width//len(frequency))]
    sample_rate, duration = 44100, 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 32767 * np.sin(2 * np.pi * freq_to_play * t)
    stereo_wave = np.column_stack((wave, wave))
    sound = pygame.sndarray.make_sound(stereo_wave.astype(np.int16))
    sound.play()
    pygame.time.delay(int(duration * 1000))


cap = cv2.VideoCapture(0)

def draw_grid(img, cols, color=(0, 0, 0), thickness=1):
    h, w = img.shape[:2]

    for j in range(0, w, w // cols):
        cv2.line(img, (j, 0), (j, h),  color, thickness)

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
            index_finger_tip = hand_landmarks.landmark[12]
            x = int(index_finger_tip.x * frame.shape[1])  # x-coordinate (horizontal)
            y = int(index_finger_tip.y * frame.shape[0])  # y-coordinate (vertical)
            
            # Draw the hand landmarks on the frame for visualization
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Play note and modulate pitch based on hand position
            play_note_and_modulate_pitch(x, y, frame.shape[1], frame.shape[0],frequency)
    
    # Show the camera feed
    fframe=cv2.flip(frame,1)
    draw_grid(fframe, len(frequency)-1)
    cv2.imshow('Hand Tracking', fframe)

    if cv2.waitKey(5) & 0xFF == 27:  # Exit if the 'Esc' key is pressed
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
