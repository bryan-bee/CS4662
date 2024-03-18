import tensorflow as tf

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Number of GPUs available:", len(gpus))
    for gpu in gpus:
        print("GPU name:", gpu.name)
else:
    print("No GPU available, using CPU.")

print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
