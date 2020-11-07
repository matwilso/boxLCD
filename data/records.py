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

def episode_example(ep, cfg):
    feature = {}
    KEYS = ['state', 'act', 'rew']
    KEYS = ['image']+KEYS if cfg.use_image else KEYS
    for key in KEYS:
        feature[key] = _bytes_feature(ep[key].astype('float32').tobytes())
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_barrel(filename, barrel_dict, cfg):
    #with tf.io.TFRecordWriter(str(filename)) as writer:
    with tf.io.TFRecordWriter(str(filename), options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
        for i in range(len(barrel_dict['state'])):
            example = episode_example(nest.map_structure(lambda x: x[i], barrel_dict), cfg)
            writer.write(example.SerializeToString())
    writer.close()

@tf.function
def vgather(args):
    """vmappable gather"""
    x, y = args
    return tf.gather(x, y)

def bptt_n(batch, cfg):
    """slice out so we only take what we are going to use for training"""
    # TODO: add support for taking a few from the same traj. we could take like 4 from the same traj to get 4x the batch size for cheap
    mult = cfg['bs:mult']
    bs, nseq = tf.shape(batch['act'])[0], tf.shape(batch['act'])[1]
    shape = tf.stack([bs, cfg['sample:bptt_n']], axis=0)
    start_idxs = tf.cast(tf.random.uniform([bs*mult], 0, tf.cast(nseq-cfg['sample:bptt_n'], tf.float32)), tf.int32)
    for key in batch:
        extra = int(key == 'state' or key == 'image')
        tot = cfg['sample:bptt_n'] + extra
        #idxs = (start_idxs[:,None] + tf.range(tot)[None])
        idxs = tf.reshape(start_idxs[:,None] + tf.range(tot)[None], [bs, mult, tot])
        #idxs = tf.reshape((start_idxs[:,None] + tf.range(tot)[None]), [-1])
        batch[key] = tf.vectorized_map(vgather, (batch[key], idxs))
        batch[key] = tf.reshape(batch[key], [bs*mult, tot, -1])
    return batch

def parse_single(example_proto, state_shape, image_shape, act_n, cfg):
    size = cfg.bl
    shapes = {
        'state': [size, *state_shape],
        'act': [size, act_n],
        'rew': [size],
    }
    if cfg.use_image: shapes['image'] = [size+1, *image_shape]
    feature_description = {key: tf.io.FixedLenFeature([], tf.string) for key in shapes}
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    out = {}
    for key in shapes:
        out[key] = tf.reshape(tf.io.decode_raw(parsed[key], tf.float32), shapes[key])
    return out

def parse(example_proto, state_shape, image_shape, act_n, cfg):
    size = cfg.bl
    bs = -1
    shapes = {
        'state': [bs, size, *state_shape],
        'act': [bs, size, act_n],
        'rew': [bs, size],
    }
    if cfg.use_image: shapes['image'] = [bs, size+1, *image_shape]
    feature_description = {key: tf.io.FixedLenFeature([], tf.string) for key in shapes}
    parsed = tf.io.parse_example(example_proto, feature_description)
    out = {}
    for key in shapes:
        out[key] = tf.reshape(tf.io.decode_raw(parsed[key], tf.float32), shapes[key])
    return out

def make_dataset(barrel_path, state_shape, image_shape, act_n, cfg, repeat=True, shuffle=True, files=None):
    setup = time.time()
    num_per_barrel = cfg.bl * cfg.num_eps
    num_files = cfg.replay_size // num_per_barrel
    if files is None:
        files = list(sorted(map(lambda x: str(x), pathlib.Path(barrel_path).glob('*.tfrecord'))))[:num_files]
        onp.random.shuffle(files)
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=32)
    dataset = dataset.repeat() if repeat else dataset
    dataset = dataset.shuffle(1000) if shuffle else dataset
    dataset = dataset.batch(cfg.bs)
    dataset = dataset.map(lambda x: parse(x, state_shape, image_shape, act_n, cfg), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.map(lambda x: bptt_n(x, cfg), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(10)
    dataset = tfds.as_numpy(dataset)
    print(f'DATASET SETUP dt {time.time()-setup} num_files {len(files)} num_total {len(files)*num_per_barrel:.2e}')
    itr = iter(dataset)
    #next(itr)
    return itr

if __name__ == '__main__':
    slice_it = lambda i, arr: nest.map_structure(lambda x: x[i], arr)
    if False:
        files = pathlib.Path('logs/trash/main.py/custom-walking/-2020-10-07-18-09-41/barrels/').glob('*.npz')
        for file in files:
            arr = onp.load(file)
            arr = {key: arr[key] for key in arr.keys()}
            with tf.io.TFRecordWriter('barrels/'+file.with_suffix('.tfrecord').name, options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
                for i in range(len(arr['state'][0])):
                    example = episode_example(slice_it(i, arr))
                    writer.write(example.SerializeToString())
            writer.close()
    if True:
        cfg = {'bs': 1024, 'sample:bptt_n': 5, 'barrel_path': './barrels', 'num_per_barrel': 5000}
        itr = make_dataset(cfg)
        out = next(itr)
        start = time.time()
        for i in range(100):
            out = next(itr)
            out = nest.map_structure(lambda x: jnp.array(x), out)
        print('100', (time.time()-start) / 100)