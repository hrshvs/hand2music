import cv2
import numpy as np
import pygame
import mediapipe as mp
import multiprocessing as mp_proc

def draw_grid(img, cols, color=(0, 0, 0), thickness=1):
    h, w = img.shape[:2]

    for j in range(0, w, w // cols):
        cv2.line(img, (j, 0), (j, h),  color, thickness)

def note_frequency(octave_range,frequency):
    i=0
    for octave in octave_range:
        for pitch in range(12):
            frequency[i]=round(440*2**((octave-4)+(pitch-9)/12),2)
            i+=1

# Function to generate sound based on finger coordinates
def music_generator(coord_queue,frequency):
    pygame.mixer.init()
    sample_rate, duration = 44100, 1.0

    while True:
        # Get the latest finger coordinates
        if not coord_queue.empty():
            x, y, frame_width = coord_queue.get()  # Get x, y coordinates from the queue
            freq_to_play=frequency[min(x//(frame_width//len(frequency)),len(frequency)-1)]  # Example frequency based on x coordinate

            # Generate sine wave for the current frequency
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = 32767 * np.sin(2 * np.pi * freq_to_play * t)
            
            # Convert 1D wave to 2D (stereo)
            stereo_wave = np.column_stack((wave, wave)).astype(np.int16)

            #stop the previous sound
            try:
                sound.stop()
            except UnboundLocalError:
                pass

            # Create sound object and play
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.play()
                    
        #Adjust the sleep to control the frequency change rate
        # pygame.time.delay(200)  # Play for a short time

# Function to capture camera input and process finger coordinates
def camera_input(coord_queue,frequency):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for mediapipe hand tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get middle finger tip coordinates (landmark 12)
                x = int(hand_landmarks.landmark[12].x * frame.shape[1])  # x-coordinate
                y = int(hand_landmarks.landmark[12].y * frame.shape[0])  # y-coordinate
                frame_width=frame.shape[1]

                # Send the coordinates to the music generator process
                coord_queue.put((x, y,frame_width))

                # Draw hand landmarks for visualization
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the camera feed with hand landmarks
        fframe=cv2.flip(frame,1)
        draw_grid(fframe, len(frequency[:])-1)
        cv2.imshow('Hand Tracking', fframe)

        if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'Esc' key
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to start both subprocesses
if __name__ == '__main__':
    octave_range=range(3,5)

    coord_queue = mp_proc.Queue()  # Shared queue for communication between processes
    frequency=mp_proc.Array('d',len(octave_range)*12)

    note_frequency(octave_range,frequency)
    # Create the two subprocesses
    p_music = mp_proc.Process(target=music_generator, args=(coord_queue,frequency))
    p_camera = mp_proc.Process(target=camera_input, args=(coord_queue,frequency))

    # Start the subprocesses
    p_music.start()
    p_camera.start()

    # Wait for both subprocesses to finish (this will keep the main program running)
    p_camera.join()
    p_music.terminate()  # Terminate the music generator when camera is done
