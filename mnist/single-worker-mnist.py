import mnist
from timeit import default_timer as timer


start = timer()
batch_size = 64
single_worker_dataset = mnist.mnist_dataset(batch_size)
single_worker_model = mnist.build_and_compile_cnn_model()
single_worker_model.summary()
single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)
single_worker_model.save("single_worker_mnist_local")
end = timer()
print(f"Time for whole process= {end-start} sec")