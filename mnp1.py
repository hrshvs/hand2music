import cv2
import numpy as np
import pygame
import mediapipe as mp
import multiprocessing as mp_proc

def music_generator(coord_queue, stop_event):
    pygame.mixer.init(frequency=44100, size=-16, channels=1)  # Mono audio output
    
    sample_rate = 44100
    duration = 1.0  # Length of sound buffer in seconds
    base_frequency = 440  # A4 note as the base frequency
    sound_playing = False
    
    while not stop_event.is_set():
        frequency = base_frequency
        
        if not coord_queue.empty():
            x, _ = coord_queue.get()
            frequency = 440 + (x % 500)  # Modulate frequency based on x-coordinate
            
            # Generate a sine wave at the calculated frequency
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = 32767 * np.sin(2 * np.pi * frequency * t)
            wave = wave.astype(np.int16)  # Convert to 16-bit PCM format
            
            # Convert 1D wave into 2D array (for mono sound)
            wave_2d = np.column_stack([wave, wave])  # Same sound in both left and right
            
            sound = pygame.sndarray.make_sound(wave_2d)
            
            if not sound_playing:
                sound.play(loops=-1)  # Start playing if not already playing
                sound_playing = True
            else:
                sound.stop()  # Stop current sound to avoid overlap
                sound.play(loops=-1)  # Update to the new frequency
        else:
            if sound_playing:
                pygame.mixer.stop()  # Stop all sounds if no hand is detected
                sound_playing = False
        
        pygame.time.delay(100)

def camera_input(coord_queue, stop_event):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the x-coordinate of the index finger tip (landmark 8)
                x = int(hand_landmarks.landmark[8].x * frame.shape[1])
                y = int(hand_landmarks.landmark[8].y * frame.shape[0])
                
                # Send the coordinates to the music generator process
                coord_queue.put((x, y))
                
                # Draw the landmarks on the frame for visualization
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing 'Esc'
            stop_event.set()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    coord_queue = mp_proc.Queue()
    stop_event = mp_proc.Event()

    # Start the music generator and camera input as subprocesses
    p_music = mp_proc.Process(target=music_generator, args=(coord_queue, stop_event))
    p_camera = mp_proc.Process(target=camera_input, args=(coord_queue, stop_event))

    p_music.start()
    p_camera.start()

    # Wait for the camera process to finish
    p_camera.join()
    
    # Signal the music process to stop and wait for it to finish
    stop_event.set()
    p_music.join()
