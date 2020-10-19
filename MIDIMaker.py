import argparse
from midiutil import MIDIFile
from DataLoader import bach_chorales

parser = argparse.ArgumentParser('Make MIDI files')
parser.add_argument('-f', '--file', dest='midi_filename', type=str, help='File to write to', required=True)
parser.add_argument('-t', '--track', action='append', dest='tracks', default=[-1], type=int, help='Tracks to make midi files for, 0-99, -1 will do all')
args = parser.parse_args()

if -1 in args.tracks:
    tracks = bach_chorales
else:
    tracks = bach_chorales[args.tracks]

for i, track in enumerate(tracks):
    retval = MIDIFile(1)
    retval.addTempo(0, 0, 600)
    for note in track:
        retval.addNote(0, 0, note[1], note[0], note[2], 100)
        
    with open(f'{args.midi_filename}-{i}.midi', 'wb') as output_file:
        retval.writeFile(output_file)