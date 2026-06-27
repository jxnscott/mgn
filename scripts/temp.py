import tensorflow as tf


ds = tf.data.TFRecordDataset("data/flag_simple/train.tfrecord")

for record in ds.take(2):
    print(record)
