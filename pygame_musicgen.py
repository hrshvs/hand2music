import pygame
import numpy as np

pygame.mixer.init()
sample_rate, duration, frequency = 44100, 5.0, 602.0
t = np.linspace(0, duration, int(sample_rate * duration), False)
wave = 32767 * np.sin(2 * np.pi * frequency * t)
stereo_wave = np.column_stack((wave, wave))
sound = pygame.sndarray.make_sound(stereo_wave.astype(np.int16))
sound.play()
pygame.time.delay(int(duration * 1000))
pygame.mixer.quit()
