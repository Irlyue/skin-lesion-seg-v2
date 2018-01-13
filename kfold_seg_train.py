import os
import inputs
import my_utils
import seg_model
import seg_train
import tensorflow as tf


logger = my_utils.get_default_logger()


def kfold_training():
    logger.info('K-fold training...')
    config = my_utils.load_config()
    dermis = inputs.load_raw_data('dermis', config)
    dermquest = inputs.load_raw_data('dermquest', config)
    n_folds = config['n_folds']

    for i in range(n_folds):
        kfold_data = inputs.get_kth_fold(dermquest, i, n_folds, seed=config['split_seed'])
        train_data = dermis + kfold_data

        kfold_config = my_utils.get_config_for_kfold(config,
                                                     train_dir=os.path.join(config['train_dir'], str(i)),
                                                     n_examples_for_train=len(train_data))
        logger.info('Training for %i-th fold data...' % i)
        train_one_fold(train_data, kfold_config)
    logger.info('Done training')


def train_one_fold(data, config):
    n_epochs_for_train = config['n_epochs_for_train']
    n_steps_for_train = my_utils.calc_training_steps(n_epochs_for_train,
                                                     config['batch_size'],
                                                     config['n_examples_for_train'])
    my_utils.delete_if_exists(config['train_dir'])

    logger.info('Train one fold for %d steps, %d examples in total.' % (n_steps_for_train, len(data)))
    with tf.Graph().as_default() as g:
        image_ph, label_ph = seg_model.model_placeholders(config)

        def build_feed_dict(image_, label_):
            return {image_ph: image_, label_ph: label_}

        global_step = tf.train.get_or_create_global_step()
        mm = seg_model.SegModel(image_ph, is_training=True)
        train_op, summary_op, debug = seg_train.build_train(mm, label_ph, config)

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
    kfold_training()
