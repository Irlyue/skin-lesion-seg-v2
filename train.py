import model
import utils
import inputs
import tensorflow as tf
import tensorflow.contrib.slim as slim

logger = utils.get_default_logger()


def build_train(net, gt_cls_label, gt_bbox, config):
    bbox_loss = tf.reduce_sum(utils.huber_loss(tf.cast(gt_bbox - net.endpoints['bbox'], dtype=tf.float32)),
                              name='bbox_loss')
    bbox = net.endpoints['bbox']
    resized_label = utils.resize_label(gt_cls_label, config['input_size'])
    x, y, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
    roi_label = tf.image.crop_to_bounding_box(resized_label, x, y, h, w)
    seg_loss = tf.losses.sparse_softmax_cross_entropy(logits=net.endpoints['up_score'],
                                                      labels=roi_label)
    seg_loss = tf.multiply(config['lambda'], seg_loss, name='seg_loss')
    total_loss = tf.add(bbox_loss, seg_loss, name='total_loss')

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summaries.add(tf.summary.scalar('loss/bbox_loss', bbox_loss))
    summaries.add(tf.summary.scalar('loss/seg_loss', seg_loss))
    summaries.add(tf.summary.scalar('loss/total_loss', total_loss))

    solver = tf.train.AdamOptimizer(config['learning_rate'])
    train_op = solver.minimize(total_loss, global_step=tf.train.get_or_create_global_step())
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    return train_op, summary_op


def train_from_scratch():
    logger.info('Training from scratch...')
    config = utils.load_config()
    n_steps_for_train = utils.calc_training_steps(config['n_epochs_for_train'],
                                                  config['batch_size'],
                                                  config['n_examples_for_train'])
    image_ph, label_ph, bbox_ph = model.model_placeholder()

    def build_feed_dict(image, label, bbox):
        return {image_ph: image, label_ph: label, bbox_ph: bbox}

    def train_step_fn(_sess, _train_op, _global_step, kargs):
        n_examples = config['n_examples_for_train']
        idx = _sess.run(_global_step) % n_examples
        image, label, bbox = data[idx]
        feed_dict = build_feed_dict(image, label, bbox)
        _sess.run(_train_op, feed_dict=feed_dict)

    mm = model.Model(image_ph, config['input_size'])
    train_op, summary_op = build_train(mm, label_ph, bbox_ph, config)
    data = inputs.load_training_data()
    last_loss = slim.learning.train(train_op,
                                    logdir=config['train_dir'],
                                    summary_op=summary_op,
                                    train_step_fn=train_step_fn,
                                    log_every_n_steps=config['log_every'],
                                    save_summaries_secs=config['save_summaries_secs'],
                                    number_of_steps=n_steps_for_train)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    train_from_scratch()