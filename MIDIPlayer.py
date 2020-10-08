from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import argparse
import pygame

def play_music(filename):
    clock = pygame.time.Clock()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)

parser = argparse.ArgumentParser('Play MIDI files')
parser.add_argument('-f', '--file', dest='midi_filename', type=str, help='File to play', required=True)
args = parser.parse_args()

freq = 44100
bitsize = -16
channels = 2
buffer = 1024
pygame.mixer.init(freq, bitsize, channels, buffer)

pygame.mixer.music.set_volume(1)

try:
    play_music(args.midi_filename)
except KeyboardInterrupt:
    pygame.mixer.music.fadeout(1000)
    pygame.mixer.music.stop()
    raise SystemExit