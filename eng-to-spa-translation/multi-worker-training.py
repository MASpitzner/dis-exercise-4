import json
import os

import tensorflow as tf

import base
from timeit import default_timer as timer

start = timer()
per_worker_batch_size = 32
# tf_config = json.loads(os.environ["TF_CONFIG"])
# num_workers = len(tf_config["cluster"]["worker"])
# strategy = tf.distribute.MultiWorkerMirroredStrategy()
num_workers = 4
strategy = tf.distribute.MirroredStrategy()
global_batch_size = per_worker_batch_size * num_workers
train_ds, val_ds = base.eng_to_span_dataset(global_batch_size)

with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = base.build_and_compile_transformer_model()

multi_worker_model.fit(train_ds, epochs=10, validation_data=val_ds)
multi_worker_model.save_weights("weights")

end = timer()
print(f"transformer-multi-gpu;{end - start} sec")
