import my_utils
import argparse
import seg_model
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--n_gpus', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_runs', type=int, default=10)
logger = my_utils.get_default_logger()
INPUT_SIZE = [600, 400]


class RunningModel:
    def __init__(self, batch_size, input_size, gpu_count):
        with tf.Graph().as_default() as g:
            config = {'batch_size': batch_size, 'input_size': input_size}
            image_ph, label_ph = seg_model.model_placeholders(config)
            self.batch_size = batch_size
            self.input_size = input_size
            self.image_ph = image_ph
            self.net = seg_model.SegModel(image_ph, True)
            logger.info('\n******************* Patch Model ***********************\n%r\n' % self.net)
            self.sess = tf.Session(graph=g, config=tf.ConfigProto(device_count={'GPU': gpu_count}))
            self.sess.run(tf.global_variables_initializer())

    def pre_run_n_step(self, n=10):
        logger.info('Pre-run %d steps to warm up the CPU...' % n)
        with my_utils.Timer() as timer:
            for _ in range(n):
                images = np.random.randint(255, size=(self.batch_size, *self.input_size, 3), dtype=np.uint8)
                self.inference_prob(images)
        logger.info('Done in %.4fs(%.4fsecs per run)' % (timer.eclipsed, timer.eclipsed / n))

    def run_n_step(self, n_runs=100, same_image=True):
        logger.info('Run %d steps...' % n_runs)
        images = np.random.randint(255, size=(self.batch_size, *self.input_size, 3), dtype=np.uint8)
        with my_utils.Timer() as timer:
            for _ in range(n_runs):
                if not same_image:
                    images = np.random.randint(255, size=(self.batch_size, *self.input_size, 3), dtype=np.uint8)
                self.inference_prob(images)
        logger.info('Done in %.4fs(%.4fsecs per run)' % (timer.eclipsed, timer.eclipsed / n_runs))

    def _build_feed_dict(self, images):
        return {self.image_ph: images}

    def inference(self, images, ops):
        if type(ops[0]) == str:
            ops = [self.net.endpoints[op] for op in ops]
        return self.sess.run(ops, feed_dict=self._build_feed_dict(images))

    def inference_prob(self, images):
        return self.inference(images, ['prob'])[0]


def test_forward_time():
    mm = RunningModel(FLAGS.batch_size, INPUT_SIZE, gpu_count=FLAGS.n_gpus)
    mm.pre_run_n_step()
    mm.run_n_step(n_runs=FLAGS.n_runs)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    logger.info('************************************************************************')
    logger.info(FLAGS)
    logger.info('************************************************************************')
    test_forward_time()