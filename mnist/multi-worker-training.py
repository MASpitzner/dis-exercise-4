import json
import os

import tensorflow as tf

import mnist

per_worker_batch_size = 32
#tf_config = json.loads(os.environ["TF_CONFIG"])
#num_workers = len(tf_config["cluster"]["worker"])
num_workers=4
#strategy = tf.distribute.MultiWorkerMirroredStrategy()
#tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1","/gpu:2","/gpu:3"])
strategy = tf.distribute.MirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist.mnist_dataset(global_batch_size)

with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = mnist.build_and_compile_cnn_model()


multi_worker_model.fit(multi_worker_dataset, epochs=10, steps_per_epoch=70)
