import os
import math
import model
import my_utils
import inputs
import bbox_model
import tensorflow as tf
import tensorflow.contrib.slim as slim

logger = my_utils.get_default_logger()


def build_train(net, gt_cls_label, gt_bbox, config):
    logger.info("Building training operations...")
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    global_step = tf.train.get_or_create_global_step()

    transformed_gt_bbox = my_utils.reversed_bbox_transform(gt_bbox, config['input_size'][0])
    huber_losses = my_utils.huber_loss(tf.cast(transformed_gt_bbox - net.endpoints['bbox_fc'], dtype=tf.float32))
    bbox_losses = tf.reduce_sum(huber_losses, axis=1)
    bbox_loss = tf.reduce_mean(bbox_losses, name='bbox_loss')

    bbox_images = my_utils.draw_bbox(net.endpoints['images'], net.endpoints['bbox_fc'])
    summaries.add(tf.summary.image('prediction/bbox', bbox_images))

    gt_cls_label = tf.expand_dims(gt_cls_label, axis=-1)
    label_for_summary = tf.cast(gt_cls_label, tf.float32)
    summaries.add(tf.summary.image('gt/images', net.endpoints['images']))
    summaries.add(tf.summary.image('gt/labels', tf.cast(label_for_summary, tf.float32)))

    bbox = net.endpoints['bbox']
    # roi_label = utils.crop_bbox(gt_cls_label, bbox, limit=config['input_size'])
    # summaries.add(tf.summary.image('roi/labels', tf.cast(roi_label, dtype=tf.float32)))

    if config['lambda'] is None:
        seg_loss = tf.constant(0.0, dtype=tf.float32, name='seg_loss')
    else:
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

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_average_op = loss_averages.apply([total_loss, bbox_loss, seg_loss])
    for l in [total_loss, bbox_loss, seg_loss]:
        summaries.add(tf.summary.scalar('raw/' + l.op.name, l))
        summaries.add(tf.summary.scalar('avg/' + l.op.name, loss_averages.average(l)))

    summaries.add(tf.summary.scalar('learning_rate', lr))

    solver = tf.train.AdamOptimizer(lr)
    grads = solver.compute_gradients(total_loss)
    apply_gradient_op = solver.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op, loss_average_op]):
        train_op = tf.no_op('train_op')

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        summaries.add(tf.summary.histogram(var.op.name, var))

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            summaries.add(tf.summary.histogram(var.op.name + '/gradients', grad))

    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    debug = {
        'total_loss': total_loss,
        'bbox_loss': bbox_loss,
    }
    return train_op, summary_op, debug


def train_from_scratch():
    logger.info('Training from scratch...')
    config = my_utils.load_config()
    # data = inputs.load_raw_data(config['database'], config)
    # dermis_data = inputs.load_raw_data('dermis', config)
    # data = data + dermis_data
    dermis = inputs.load_raw_data('dermis', config)
    dermquest = inputs.load_raw_data('dermquest', config)
    kfold_train_data = inputs.get_kth_fold(dermquest, 0, config['n_folds'],
                                           seed=config['split_seed'])
    data = dermis + kfold_train_data

    n_examples_for_train = len(data)
    n_steps_for_train = my_utils.calc_training_steps(config['n_epochs_for_train'], config['batch_size'],
                                                     n_examples_for_train)

    config['n_examples_for_train'] = n_examples_for_train
    with tf.Graph().as_default() as g:
        image_ph, label_ph, bbox_ph = bbox_model.model_placeholder(config)

        def build_feed_dict(image_, label_, bbox_):
            return {image_ph: image_, label_ph: label_, bbox_ph: bbox_}

        global_step = tf.train.get_or_create_global_step()
        mm = bbox_model.Model(image_ph, config['input_size'])
        train_op, summary_op, debug = build_train(mm, label_ph, bbox_ph, config)

        logger.info('Done loading data set `%s`, %i examples in total' % (config['database'], len(data)))

        my_utils.create_and_delete_if_exists(config['train_dir'])
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(config['train_dir'], graph=g)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i, (images, labels, bboxes) in enumerate(data.aug_train_batch(config)):
                # image, label, bbox = data[0]
                feed_dict = build_feed_dict(images, labels, bboxes)
                ops = [debug['bbox_loss'], debug['total_loss'], train_op]
                bbox_loss_val, total_loss_val, _ = sess.run(ops, feed_dict=feed_dict)
                if i % config['log_every'] == 0:
                    fmt = 'step {:>5}/{} bbox_loss {:.5f}, total_loss {:.5f}'
                    logger.info(fmt.format(i, n_steps_for_train, bbox_loss_val, total_loss_val))

                if i % config['checkpoint_every'] == 0:
                    my_utils.save_model(saver, config)
                    logger.info('Model saved at step-%i' % sess.run(global_step))

                if config['save_summary_every'] and i % config['save_summary_every'] == 0:
                    my_utils.add_summary(writer, summary_op, feed_dict)
                    logger.info('Summary saved at step-%i' % sess.run(global_step))

            save_path = my_utils.save_model(saver, config)
            logger.info('Done training, model saved at %s' % (save_path,))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    train_from_scratch()