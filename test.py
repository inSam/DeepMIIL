import tensorflow as tf
import numpy as np
import os
import glob

input_dir = "temp"
input_paths = glob.glob(os.path.join(input_dir, "*.bin"))
def get_name(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name

# if the image names are numbers, sort by the value rather than asciibetically
# having sorted inputs means that the outputs are sorted in test mode
if all(get_name(path).isdigit() for path in input_paths):
    input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
else:
    input_paths = sorted(input_paths)

with tf.Session() as sess:
    
    path_queue = tf.train.string_input_producer(input_paths, True)
    reader = tf.FixedLengthRecordReader(2323200);
    paths, contents = reader.read(path_queue)
    print(contents.get_shape())
