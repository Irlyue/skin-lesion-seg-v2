import os
import math
import model
import utils
import inputs
import tensorflow as tf
import tensorflow.contrib.slim as slim

logger = utils.get_default_logger()


def build_train(net, gt_cls_label, gt_bbox, config):
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    global_step = tf.train.get_or_create_global_step()
    transformed_gt_bbox = utils.reversed_bbox_transform(gt_bbox, config['input_size'][0])
    bbox_loss = tf.reduce_sum(utils.huber_loss(tf.cast(transformed_gt_bbox - net.endpoints['fc6'], dtype=tf.float32)),
                              name='bbox_loss')

    gt_cls_label = tf.expand_dims(gt_cls_label, axis=0)
    gt_cls_label = tf.expand_dims(gt_cls_label, axis=-1)
    label_for_summary = tf.cast(gt_cls_label, tf.float32)
    summaries.add(tf.summary.image('gt/images', net.endpoints['images']))
    summaries.add(tf.summary.image('gt/labels', tf.cast(label_for_summary, tf.float32)))
    if config['lambda'] is None:
        seg_loss = 0.0
    else:
        bbox = net.endpoints['bbox']
        roi_label = utils.crop_bbox(gt_cls_label, bbox, limit=config['input_size'])
        summaries.add(tf.summary.image('roi/labels', tf.cast(roi_label, dtype=tf.float32)))
        summaries.add(tf.summary.image('roi/predictions', tf.cast(net.endpoints['lesion_mask'], dtype=tf.float32)))
        seg_loss = tf.losses.sparse_softmax_cross_entropy(logits=net.endpoints['roi'],
                                                          labels=roi_label)
        seg_loss = tf.multiply(config['lambda'], seg_loss, name='seg_loss')

    total_loss = tf.add(bbox_loss, seg_loss, name='total_loss')

    n_steps_per_epoch = int(math.ceil(config['n_examples_for_train'] // config['batch_size']))
    n_epochs_per_decay = config['n_epochs_per_decay']
    decay_steps = n_epochs_per_decay * n_steps_per_epoch
    lr = tf.train.exponential_decay(config['learning_rate'],
                                    global_step=global_step,
                                    decay_rate=config['lr_decay_rate'],
                                    decay_steps=decay_steps,
                                    staircase=True)

    summaries.add(tf.summary.scalar('loss/bbox_loss', bbox_loss))
    summaries.add(tf.summary.scalar('loss/seg_loss', seg_loss))
    summaries.add(tf.summary.scalar('loss/total_loss', total_loss))
    summaries.add(tf.summary.scalar('learning_rate', lr))

    solver = tf.train.AdamOptimizer(lr)
    grads = solver.compute_gradients(total_loss)
    apply_gradient_op = solver.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op('train_op')

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        summaries.add(tf.summary.histogram(var.op.name, var))

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            summaries.add(tf.summary.histogram(var.op.name + '/gradients', grad))

    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    debug = {'total_loss': total_loss}
    return train_op, summary_op, debug


def train_from_scratch():
    with tf.Graph().as_default() as g:
        logger.info('Training from scratch...')
        config = utils.load_config()
        n_steps_for_train = utils.calc_training_steps(config['n_epochs_for_train'],
                                                      config['batch_size'],
                                                      config['n_examples_for_train'])
        image_ph, label_ph, bbox_ph = model.model_placeholder(config)

        def build_feed_dict(image, label, bbox):
            return {image_ph: image, label_ph: label, bbox_ph: bbox}

        global_step = tf.train.get_or_create_global_step()
        mm = model.Model(image_ph, config['input_size'], config['roi_size'])
        train_op, summary_op, debug = build_train(mm, label_ph, bbox_ph, config)
        data = inputs.load_training_data('dermis', config)

        utils.create_and_delete_if_exists(config['train_dir'])
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(config['train_dir'], graph=g)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(n_steps_for_train):
                image, label, bbox = data[i % config['n_examples_for_train']]
                # image, label, bbox = data[0]
                feed_dict = build_feed_dict(image, label, bbox)
                loss_val, bbox_val, _ = sess.run([debug['total_loss'], mm.endpoints['bbox'], train_op], feed_dict=feed_dict)
                if i % config['log_every'] == 0:
                    logger.info('step %i, loss %.3f' % (i, loss_val))
                    logger.info('bbox_gt %r, bbox_pred %r' % (bbox, bbox_val))

                if i % config['checkpoint_every'] == 0:
                    utils.save_model(saver, config)

                if i % config['save_summary_every'] == 0:
                    utils.add_summary(writer, summary_op, feed_dict)
            save_path = utils.save_model(saver, config)
            logger.info('Done training, model saved at %s' % (save_path,))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    train_from_scratch()