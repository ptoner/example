import tensorflow as tf
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned.linear import LinearClassifier
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer

device_name = "/gpu:0"

def train():

    tf.logging.set_verbosity(tf.logging.INFO)


    with tf.device(device_name):

        train_input_fn = create_train_input_function(
            train_tf_records_filename="train.tfrecords",
            batch_size=128,
            shuffle_while_training=True,
            shuffle_buffer_size=1000,
            train_epochs=10000,
            device_name=device_name
        )


        eval_input_fn = create_eval_input_function(
            eval_tf_records_filename="eval.tfrecords",
            device_name=device_name
        )

        estimator = LinearClassifier(

            config = tf.estimator.RunConfig(
                save_checkpoints_secs=360,
                model_dir="model",
                session_config=tf.ConfigProto(
                    log_device_placement=False,
                    allow_soft_placement=False,
                )
            ),
            optimizer=FtrlOptimizer(
                learning_rate= 0.001,
                l1_regularization_strength= 0.001,
                l2_regularization_strength= 0.001
            ),

            feature_columns=[
                tf.feature_column.numeric_column("projectedPoints")
            ]
        )


        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            max_steps=100000000
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            throttle_secs=1000,
            start_delay_secs=1000
        )

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def parser(serialized_example):
    features = {}

    features['pickTier'] = tf.FixedLenFeature([], tf.int64)
    features['projectedPoints'] = tf.FixedLenFeature([], tf.float32)


    parsed_example = tf.parse_single_example(
        serialized_example,
        features=features)


    label = parsed_example["pickTier"]

    # Remove label
    parsed_example.pop("pickTier", None)


    return parsed_example, label




def create_train_input_function(train_tf_records_filename=None, batch_size=None, shuffle_while_training=None, shuffle_buffer_size=None, train_epochs=None, device_name=None):

    def train_input_fn():

        # with tf.device("/cpu:0"):
        filenames = [train_tf_records_filename]

        dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')

        def the_parser(serialized_example):
            return parser(serialized_example)

        # Map the parser over dataset, and batch results by up to batch_size
        dataset = dataset.map(the_parser)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(train_epochs)

        if shuffle_while_training:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)


        dataset = dataset.apply(
            tf.contrib.data.prefetch_to_device(
                device_name
            )
        )

        iterator = dataset.make_one_shot_iterator()

        features, labels = iterator.get_next()

        return features, labels

    return train_input_fn


def create_eval_input_function(eval_tf_records_filename, device_name):

    def eval_input_fn():

        # with tf.device("/cpu:0"):

        filenames = [eval_tf_records_filename]
        dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')

        def the_parser(serialized_example):
            return parser(serialized_example)


        dataset = dataset.map(the_parser)
        dataset = dataset.batch(18814)

        iterator = dataset.make_one_shot_iterator()

        features, labels = iterator.get_next()

        return features, labels

    return eval_input_fn


train()