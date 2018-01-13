import math
import inputs
import my_utils
import seg_model
import tensorflow as tf

logger = my_utils.get_default_logger()


def build_train(net, labels, config):
    logger.info('Building training operations...')

    global_step = tf.train.get_or_create_global_step()
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # [1, None, None, 2]
    logits = net.endpoints['up_score']
    seg_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, scope='seg_loss')
    if config['reg']:
        reg = tf.constant(config['reg'], dtype=tf.float32, name='reg')
        reg_loss = tf.multiply(reg, tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name='reg_loss')
        total_loss = tf.add(reg_loss, seg_loss, name='total_loss')
        summaries.add(tf.summary.scalar('loss/reg_loss', reg_loss))
    else:
        total_loss = tf.add(seg_loss, 0.0, name='total_loss')

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_average_op = loss_averages.apply([total_loss, seg_loss])
    for l in [total_loss, seg_loss]:
        summaries.add(tf.summary.scalar('raw/' + l.op.name, l))
        summaries.add(tf.summary.scalar('avg/' + l.op.name, loss_averages.average(l)))

    # label summary
    expanded_labels = tf.expand_dims(labels, axis=-1)
    cast_labels = tf.cast(expanded_labels, dtype=tf.float32)
    summaries.add(tf.summary.image('mask/label', cast_labels))
    # prediction summary
    expanded_pred = tf.expand_dims(net.endpoints['mask'], axis=-1)
    cast_pred = tf.cast(expanded_pred, tf.float32)
    summaries.add(tf.summary.image('mask/prediction', cast_pred))

    # image summary
    summaries.add(tf.summary.image('images', net.endpoints['images']))

    # metric summary
    accuracy, sensitivity, specificity, update_op = my_utils.metric_summary_op(labels, net.endpoints['mask'])
    summaries.add(tf.summary.scalar('metric/accuracy', accuracy))
    summaries.add(tf.summary.scalar('metric/sensitivity', sensitivity))
    summaries.add(tf.summary.scalar('metric/specificity', specificity))

    n_steps_per_epoch = int(math.ceil(config['n_examples_for_train'] // config['batch_size']))
    n_epochs_per_decay = config['n_epochs_per_decay']
    decay_steps = n_epochs_per_decay * n_steps_per_epoch
    lr = tf.train.exponential_decay(config['learning_rate'],
                                    global_step=global_step,
                                    decay_rate=config['lr_decay_rate'],
                                    decay_steps=decay_steps,
                                    staircase=True)
    summaries.add(tf.summary.scalar('learning_rate', lr))

    solver = tf.train.AdamOptimizer(lr)
    train_op = solver.minimize(total_loss, global_step=global_step)

    with tf.control_dependencies([loss_average_op, update_op]):
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

    debug = {
        'seg_loss': seg_loss,
        'total_loss': total_loss,
        'accuracy': accuracy,
        'specificity': specificity,
        'sensitivity': sensitivity
    }
    return train_op, summary_op, debug


def train_from_scratch():
    logger.info('Training from scratch...')
    config = my_utils.load_config()

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
        image_ph, label_ph = seg_model.model_placeholders(config)

        def build_feed_dict(image_, label_):
            return {image_ph: image_, label_ph: label_}

        global_step = tf.train.get_or_create_global_step()
        mm = seg_model.SegModel(image_ph, is_training=True)
        train_op, summary_op, debug = build_train(mm, label_ph, config)

        logger.info('Done loading data set `%s`, %i examples in total' % (config['database'], len(data)))

        my_utils.create_and_delete_if_exists(config['train_dir'])
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(config['train_dir'], graph=g)
        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            for i, (images, labels, _) in enumerate(data.aug_train_batch(config)):
                feed_dict = build_feed_dict(images, labels)
                ops = [debug['seg_loss'], debug['total_loss'], debug['accuracy'], train_op]
                seg_loss_val, total_loss_val, accuracy_val, _ = sess.run(ops, feed_dict=feed_dict)
                if i % config['log_every'] == 0:
                    fmt = 'step {:>5}/{} seg_loss {:.5f}, total_loss {:.5f}, pixel accuracy: {:.3f}'
                    logger.info(fmt.format(i, n_steps_for_train, seg_loss_val, total_loss_val, accuracy_val))

                if i % config['checkpoint_every'] == 0:
                    my_utils.save_model(saver, config)
                    logger.info('Model saved at step-%i' % sess.run(global_step))

                if config['save_summary_every'] and i % config['save_summary_every'] == 0:
                    my_utils.add_summary(writer, summary_op, feed_dict)
                    logger.info('Summary saved at step-%i' % sess.run(global_step))

            save_path = my_utils.save_model(saver, config)
            logger.info('Done training, model saved at %s' % (save_path,))


if __name__ == '__main__':
    train_from_scratch()
