import json
import os

import tensorflow as tf

import mnist
from timeit import default_timer as timer


start = timer()
per_worker_batch_size = 64

#tf_config = json.loads(os.environ["TF_CONFIG"])
#num_workers = len(tf_config["cluster"]["worker"])

#strategy = tf.distribute.MultiWorkerMirroredStrategy()
num_workers = 4
strategy = tf.distribute.MirroredStrategy()
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist.mnist_dataset(global_batch_size)

with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = mnist.build_and_compile_cnn_model()


multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
end = timer()
print(f"mnist-multi-gpu;{end-start} sec")
