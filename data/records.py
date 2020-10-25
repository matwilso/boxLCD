import time
import pathlib
import io
import uuid
from datetime import datetime
import pathlib
from threading import Thread, Lock
from collections import defaultdict
import copy
from tensorflow import nest
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as onp
import jax.numpy as jnp
from mbrl import utils
A = utils.A
from jax import tree_util

land = onp.logical_and
lor = onp.logical_or
lnot = onp.logical_not


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

KEYS = ['image', 'state', 'act', 'rew']
def episode_example(ep):
    feature = {}
    for key in KEYS:
        feature[key] = _bytes_feature(ep[key].astype('float32').tobytes())
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_barrel(filename, barrel_dict):
    #with tf.io.TFRecordWriter(str(filename)) as writer:
    with tf.io.TFRecordWriter(str(filename), options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
        for i in range(len(barrel_dict['state'])):
            example = episode_example(nest.map_structure(lambda x: x[i], barrel_dict))
            writer.write(example.SerializeToString())
    writer.close()


@tf.function
def vgather(args):
    """vmappable gather"""
    x, y = args
    return tf.gather(x, y)

def bptt_n(batch, FLAGS):
    """slice out so we only take what we are going to use for training"""
    # TODO: add support for taking a few from the same traj. we could take like 4 from the same traj to get 4x the batch size for cheap
    mult = FLAGS['bs:mult']
    bs, nseq = tf.shape(batch['act'])[0], tf.shape(batch['act'])[1]
    shape = tf.stack([bs, FLAGS['sample:bptt_n']], axis=0)
    start_idxs = tf.cast(tf.random.uniform([bs*mult], 0, tf.cast(nseq-FLAGS['sample:bptt_n'], tf.float32)), tf.int32)
    for key in batch:
        if key == 'id':
            continue
        extra = int(key == 'o:state' or key == 'o:image')
        tot = FLAGS['sample:bptt_n'] + extra
        #idxs = (start_idxs[:,None] + tf.range(tot)[None])
        idxs = tf.reshape(start_idxs[:,None] + tf.range(tot)[None], [bs, mult, tot])
        #idxs = tf.reshape((start_idxs[:,None] + tf.range(tot)[None]), [-1])
        batch[key] = tf.vectorized_map(vgather, (batch[key], idxs))
        batch[key] = tf.reshape(batch[key], [bs*mult, tot, -1])
    return batch

def parse_single(example_proto, obs_n, act_n, FLAGS):
    size = FLAGS['env:timeout']
    shapes = {
        'obs': [size+1, obs_n],
        'act': [size, act_n],
        'rew': [size],
    }
    feature_description = {key: tf.io.FixedLenFeature([], tf.string) for key in KEYS}
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    out = {}
    for key in shapes:
        out[key] = tf.reshape(tf.io.decode_raw(parsed[key], tf.float32), shapes[key])
    return out

def parse(example_proto, obs_n, act_n, FLAGS):
    size = FLAGS['env:timeout']
    bs = FLAGS['bs']
    shapes = {
        'obs': [bs, size+1, obs_n],
        'act': [bs, size, act_n],
        'rew': [bs, size],
        'id': [bs] 
    }
    feature_description = {key: tf.io.FixedLenFeature([], tf.string) for key in KEYS}
    parsed = tf.io.parse_example(example_proto, feature_description)
    out = {}
    for key in shapes:
        out[key] = tf.reshape(tf.io.decode_raw(parsed[key], tf.float32), shapes[key])
    return out

def make_dataset(obs_n, act_n, FLAGS):
    setup = time.time()
    num_files = FLAGS['replay_size'] // FLAGS['num_per_barrel']
    files = list(sorted(map(lambda x: str(x), pathlib.Path(FLAGS['barrel_path']).glob('*.tfrecord'))))[:num_files]
    onp.random.shuffle(files)
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=32)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(5000)
    dataset = dataset.batch(FLAGS['bs'])
    dataset = dataset.map(lambda x: parse(x, obs_n, act_n, FLAGS), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: bptt_n(x, FLAGS), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(100)
    dataset = tfds.as_numpy(dataset)
    print(f'DATASET SETUP dt {time.time()-setup} num_files {len(files)} num_total {len(files)*FLAGS["num_per_barrel"]:.2e}')
    itr = iter(dataset)
    next(itr)
    return itr

if __name__ == '__main__':
    slice_it = lambda i, arr: nest.map_structure(lambda x: x[i], arr)
    if False:
        files = pathlib.Path('logs/trash/main.py/custom-walking/-2020-10-07-18-09-41/barrels/').glob('*.npz')
        for file in files:
            arr = onp.load(file)
            arr = {key: arr[key] for key in arr.keys()}
            with tf.io.TFRecordWriter('barrels/'+file.with_suffix('.tfrecord').name, options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
                for i in range(len(arr['obs'][0])):
                    example = episode_example(slice_it(i, arr))
                    writer.write(example.SerializeToString())
            writer.close()
    if True:
        FLAGS = {'bs': 1024, 'sample:bptt_n': 5, 'barrel_path': './barrels', 'num_per_barrel': 5000}
        itr = make_dataset(FLAGS)
        out = next(itr)
        start = time.time()
        for i in range(100):
            out = next(itr)
            out = nest.map_structure(lambda x: jnp.array(x), out)
        print('100', (time.time()-start) / 100)
