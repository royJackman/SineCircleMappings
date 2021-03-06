import argparse
import numpy as np
import pretty_midi
import sys

def midi_to_chroma(midi_file, fs=4):
    return pretty_midi.PrettyMIDI(midi_file).get_chroma(fs=fs)

def midi_to_piano_roll(midi_file, fs=4):
    return pretty_midi.PrettyMIDI(midi_file).get_piano_roll(fs=fs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert MIDI files to numpy')
    parser.add_argument('-i', '--input-file', dest='input_filename', type=str, default=None, help='File to read from')
    parser.add_argument('-o', '--output-file', dest='output_filename', type=str, default='output', help='File to save output')
    args = parser.parse_args()

    if args.input_filename is None:
        sys.exit('Need a file to convert!')
    elif '.mid' in args.input_filename or '.midi' in args.input_filename:
        np.save(args.output_filename, midi_to_chroma(args.input_filename))