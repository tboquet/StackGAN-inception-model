# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import numpy as np
import tensorflow as tf
import h5py
from tqdm import trange
import keras.backend as K
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
# from inception.keras_inception import InceptionV3
# from inception.keras_inception import preprocess_input


import math
import scipy.misc
# import time
# import scipy.io as sio
# from datetime import datetime
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint_dir',
    './inception_finetuned_models/birds_valid299/model.ckpt-5000',
    """Path where to read model checkpoints.""")

tf.app.flags.DEFINE_string(
    'image_folder', '/Users/han/Documents/CUB_200_2011/CUB_200_2011/images',
    """Path where to load the images """)

tf.app.flags.DEFINE_integer(
    'num_classes',
    50,  # 20 for flowers
    """Number of classes """)
tf.app.flags.DEFINE_integer('splits', 10, """Number of splits """)
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('gpu', 1, "The ID of GPU to use")

fullpath = FLAGS.image_folder
print(fullpath)


def preprocess(img):
    if len(img.shape) == 2:
        img = np.resize(img, (img.shape[0], img.shape[1], 3))
    img = scipy.misc.imresize(img, (299, 299, 3), interp='bilinear')
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, 0)


def get_inception_score(images, model):
    splits = FLAGS.splits
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    bs = FLAGS.batch_size
    preds = []
    num_examples = len(images)
    # TODO: remove
    num_examples = 1000
    n_batches = int(math.floor(float(num_examples) / float(bs)))
    indices = list(np.arange(num_examples))
    np.random.shuffle(indices)
    for i in trange(n_batches):
        inp = []
        # print('i*bs', i*bs)
        for j in range(bs):
            if (i * bs + j) == num_examples:
                break
            img = images[indices[i * bs + j]]
            # print('*****', img.shape)
            img = preprocess(img)
            inp.append(img)
        # print("%d of %d batches" % (i, n_batches))
        # inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        #  print('inp', inp.shape)
        pred = model.predict(inp)
        preds.append(pred)
        if i % 100 == 0:
            print('Batch ', i)
            print('inp', inp.shape, inp.max(), inp.min())
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        istart = i * preds.shape[0] // splits
        iend = (i + 1) * preds.shape[0] // splits
        part = preds[istart:iend, :]
        kl = (part *
              (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0))))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    print('mean:', "%.2f" % np.mean(scores), 'std:', "%.2f" % np.std(scores))
    return np.mean(scores), np.std(scores)


def load_data(fullpath):
    print(fullpath)
    f = h5py.File(fullpath, mode='r')
    images = f['input_image']
    return images


def main(unused_argv=None):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            K.set_session(sess)
            model = InceptionV3(
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000)

            images = load_data(fullpath)
            get_inception_score(images, model)


if __name__ == '__main__':
    tf.app.run()
