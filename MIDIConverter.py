import argparse
import numpy as np
import pretty_midi
import sys

parser = argparse.ArgumentParser('Convert MIDI files to numpy')
parser.add_argument('-i', '--input-file', dest='input_filename', type=str, default=None, help='File to read from')
parser.add_argument('-o', '--output-file', dest='output_filename', type=str, default='output', help='File to save output')
args = parser.parse_args()

def midi_to_chroma(midi_file):
    return pretty_midi.PrettyMIDI(midi_file).get_chroma()

if args.input_filename is None:
    sys.exit('Need a file to convert!')
elif '.mid' in args.input_filename or '.midi' in args.input_filename:
    np.save(args.output_filename, midi_to_chroma(args.input_filename))