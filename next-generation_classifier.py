import tensorflow as tf
import pickle
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import flatten


import skimage.data
import skimage.transform

from sklearn.utils import shuffle
import time



datasets_path = "traffic-signs-data/"


def load_traffic_sign_data(training_file, testing_file,validation_file):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    return train, test,valid

train, test, valid = load_traffic_sign_data(datasets_path + 'train.p', datasets_path + 'test.p', datasets_path + 'valid.p')

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
X_valid, y_valid = valid['features'], valid['labels']


n_classes = len(set(y_train))

sign_names = pd.read_csv("signnames.csv")
print(sign_names)

def group_img_id_to_lb_count(X_train_id_to_label):
    return pd.pivot_table(X_train_id_to_label,index=["label_id","label_name"], values=["img_id"],aggfunc='count')



def group_img_id_to_lbl(y_train,sign_names):
    arr_map = []
    for i in range(0,y_train.shape[0]):
        label_id = y_train[i]
        label_name = sign_names[sign_names["ClassId"] == label_id]["SignName"].values[0]
        arr_map.append({"img_id":i, "label_id": label_id, "label_name": label_name})
    return pd.DataFrame(arr_map)


X_train_id_to_label = group_img_id_to_lbl(y_train, sign_names)
X_train_group_by_label_count = group_img_id_to_lb_count(X_train_id_to_label)

#X_train_group_by_label_count.plot(kind='bar', figsize=(15, 7))
X_train_group_by_label = X_train_id_to_label.groupby(["label_id", "label_name"])


X_valid_id_to_label = group_img_id_to_lbl(y_valid, sign_names)
X_valid_group_by_label_count = group_img_id_to_lb_count(X_valid_id_to_label)

#X_valid_group_by_label_count.plot(kind='bar', figsize=(15, 7))
X_valid_group_by_label = X_valid_id_to_label.groupby(["label_id", "label_name"])




# def create_sample_set(grouped_imgs_by_label, imgs, labels, pct=0.4):
#     """
#     Creates a sample set containing pct elements of the original grouped dataset
#     """
#     X_sample = []
#     y_sample = []
#
#     for (lid, lbl), group in grouped_imgs_by_label:
#         group_size = group['img_id'].size
#         img_count_to_copy = int(group_size * pct)
#         rand_idx = np.random.randint(0, high=group_size, size=img_count_to_copy, dtype='int')
#
#         selected_img_ids = group.iloc[rand_idx]['img_id'].values
#         selected_imgs = imgs[selected_img_ids]
#         selected_labels = labels[selected_img_ids]
#         X_sample = selected_imgs if len(X_sample) == 0 else np.concatenate((selected_imgs, X_sample), axis=0)
#         y_sample = selected_labels if len(y_sample) == 0 else np.concatenate((selected_labels, y_sample), axis=0)
#
#
#     return (X_sample, y_sample)

#
# X_sample_train, y_sample_train = create_sample_set(X_train_group_by_label, X_train, y_train, pct=0.33)
# #print("Sample training images dimensions={0}, labels dimensions={1}".format(X_sample_train.shape, y_sample_train.shape))
#
#
# X_sample_valid, y_sample_valid = create_sample_set(X_valid_group_by_label, X_valid, y_valid, pct=0.33)
# #print("Sample validation images dimensions={0}, labels dimensions={1}".format(X_sample_valid.shape, y_sample_valid.shape))




# data pre-processing

def normalise_images(imgs, dist):
    """
    Nornalise the supplied images from data in dist
    """
    std = np.std(dist)
    #std = 128
    mean = np.mean(dist)
    #mean = 128
    return (imgs - mean) / std

def to_grayscale(img):
    """
    Converts an image in RGB format to grayscale
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


X_train_normalised = normalise_images(X_train, X_train)
X_valid_normalised = normalise_images(X_valid, X_train)
X_test_normalised = normalise_images(X_test, X_train)


X_train_grayscale = np.asarray(list(map(lambda img: to_grayscale(img), X_train)))
X_valid_grayscale = np.asarray(list(map(lambda img: to_grayscale(img), X_valid)))
X_test_grayscale = np.asarray(list(map(lambda img: to_grayscale(img), X_test)))



class ModelConfig:
    def __init__(self, model, name, input_img_dimensions, conv_layers_config, fc_output_dims, output_classes,
                 dropout_keep_pct):
        self.model = model
        self.name = name
        self.input_img_dimensions = input_img_dimensions

        # Determines the wxh dimension of filters, the starting depth (increases by x2 at every layer)
        # and how many convolutional layers the network has
        self.conv_filter_size = conv_layers_config[0]
        self.conv_depth_start = conv_layers_config[1]
        self.conv_layers_count = conv_layers_config[2]

        self.fc_output_dims = fc_output_dims
        self.output_classes = output_classes

        # Try with different values for drop out at convolutional and fully connected layers
        self.dropout_conv_keep_pct = dropout_keep_pct[0]
        self.dropout_fc_keep_pct = dropout_keep_pct[1]



class ModelExecutor:
    def __init__(self, model_config, learning_rate=0.001):
        self.model_config = model_config
        self.learning_rate = learning_rate

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with g.name_scope(self.model_config.name) as scope:
                # Create Model operations
                self.create_model_operations()

                # Create a saver to persist the results of execution
                self.saver = tf.train.Saver()

    def create_placeholders(self):
        """
        Defining our placeholder variables:
            - x, y
            - one_hot_y
            - dropout placeholders
        """

        # e.g. 32 * 32 * 3
        input_dims = self.model_config.input_img_dimensions
        self.x = tf.placeholder(tf.float32, (None, input_dims[0], input_dims[1], input_dims[2]),
                                name="x")
        self.y = tf.placeholder(tf.int32, (None), name="y")
        self.one_hot_y = tf.one_hot(self.y, self.model_config.output_classes)

        self.dropout_placeholder_conv = tf.placeholder(tf.float32,name = "drop_conv" )
        self.dropout_placeholder_fc = tf.placeholder(tf.float32,name="drop_fc")

    def create_model_operations(self):
        """

        Sets up all operations needed to execute run deep learning pipeline
        """

        # First step is to set our x, y, etc
        self.create_placeholders()

        cnn = self.model_config.model

        # Build the network -  TODO: pass the configuration in the future
        self.logits = cnn(self.x, self.model_config, self.dropout_placeholder_conv, self.dropout_placeholder_fc)
        # Obviously, using softmax as the activation function for final layer
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y, logits=self.logits)
        # Combined all the losses across batches
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        # What method do we use to reduce our loss?
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # What do we really do in a training operation then? Answer: we attempt to reduce the loss using our chosen optimizer
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        # Get the top prediction for model against labels and check whether they match
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        # Compute accuracy at batch level
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # compute what the prediction would be, when we don't have matching label
        self.prediction = tf.argmax(self.logits, 1, name= "pre")
        # Registering our top 5 predictions
        self.top5_predictions = tf.nn.top_k(tf.nn.softmax(self.logits), k=5, sorted=True, name=None)

    def evaluate_model(self, X_data, Y_data, batch_size):
        """
        Evaluates the model's accuracy and loss for the supplied dataset.
        Naturally, Dropout is ignored in this case (i.e. we set dropout_keep_pct to 1.0)
        """

        num_examples = len(X_data)
        total_accuracy = 0.0
        total_loss = 0.0
        sess = tf.get_default_session()

        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset + batch_size], Y_data[offset:offset + batch_size]

            # Compute both accuracy and loss for this batch
            accuracy = sess.run(self.accuracy_operation,
                                feed_dict={
                                    self.dropout_placeholder_conv: 1.0,
                                    self.dropout_placeholder_fc: 1.0,
                                    self.x: batch_x,
                                    self.y: batch_y
                                })
            loss = sess.run(self.loss_operation, feed_dict={
                self.dropout_placeholder_conv: 1.0,
                self.dropout_placeholder_fc: 1.0,
                self.x: batch_x,
                self.y: batch_y
            })

            # Weighting accuracy by the total number of elements in batch
            total_accuracy += (accuracy * len(batch_x))
            total_loss += (loss * len(batch_x))

            # To produce a true mean accuracy over whole dataset
        return (total_accuracy / num_examples, total_loss / num_examples)

    def train_model(self, X_train_features, X_train_labels, X_valid_features, y_valid_labels, batch_size=512,
                    epochs=100, PRINT_FREQ=10):
        """
        Trains the model for the specified number of epochs supplied when creating the executor
        """

        # Create our array of metrics
        training_metrics = np.zeros((epochs, 3))
        validation_metrics = np.zeros        ((epochs, 3))

        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train_features)

            print("Training {0} [epochs={1}, batch_size={2}]...\n".format(self.model_config.name, epochs, batch_size))

            for i in range(epochs):
                start = time.time()
                X_train, Y_train = shuffle(X_train_features, X_train_labels)
                for offset in range(0, num_examples, batch_size):
                    end = offset + batch_size
                    batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
                    sess.run(self.training_operation, feed_dict={
                        self.x: batch_x,
                        self.y: batch_y,
                        self.dropout_placeholder_conv: self.model_config.dropout_conv_keep_pct,
                        self.dropout_placeholder_fc: self.model_config.dropout_fc_keep_pct,

                    })


                end_training_time = time.time()
                training_duration = end_training_time - start

                # computing training accuracy
                training_accuracy, training_loss = self.evaluate_model(X_train_features, X_train_labels, batch_size)

                # Computing validation accuracy
                validation_accuracy, validation_loss = self.evaluate_model(X_valid_features, y_valid_labels, batch_size)

                end_epoch_time = time.time()
                validation_duration = end_epoch_time - end_training_time
                epoch_duration = end_epoch_time - start

                if i == 0 or (i + 1) % PRINT_FREQ == 0:
                    print(
                    "[{0}]\ttotal={1:.3f}s | train: time={2:.3f}s, loss={3:.4f}, acc={4:.4f} | val: time={5:.3f}s, loss={6:.4f}, acc={7:.4f}".format(
                        i + 1, epoch_duration, training_duration, training_loss, training_accuracy,
                        validation_duration, validation_loss, validation_accuracy))

                training_metrics[i] = [training_duration, training_loss, training_accuracy]
                validation_metrics[i] = [validation_duration, validation_loss, validation_accuracy]

            model_file_name = "{0}{1}.chkpt".format("lenet/", self.model_config.name)
            # Save the model
            self.saver.save(sess, model_file_name)
            print("Model {0} saved".format(model_file_name))

        return (training_metrics, validation_metrics, epoch_duration)

    def test_model(self, test_imgs, test_lbs, batch_size=512):
        """
        Evaluates the model with the test dataset and test labels
        Returns the tuple (test_accuracy, test_loss, duration)
        """

        with tf.Session(graph=self.graph) as sess:
            # Never forget to re-initialise the variables
            tf.global_variables_initializer()

            model_file_name = "{0}{1}.chkpt".format("lenet/", self.model_config.name)
            self.saver.restore(sess, model_file_name)

            start = time.time()
            (test_accuracy, test_loss) = self.evaluate_model(test_imgs, test_lbs, batch_size)
            duration = time.time() - start
            print("[{0} - Test Set]\ttime={1:.3f}s, loss={2:.4f}, acc={3:.4f}".format(self.model_config.name, duration,
                                                                                      test_loss, test_accuracy))

        return (test_accuracy, test_loss, duration)

    def predict(self, imgs, top_5=False):
        """
        Returns the predictions associated with a bunch of images
        """
        preds = None
        with tf.Session(graph=self.graph) as sess:
            # Never forget to re-initialise the variables
            tf.global_variables_initializer()

            model_file_name = "{0}{1}.chkpt".format("lenet/", self.model_config.name)
            self.saver.restore(sess, model_file_name)

            if top_5:
                preds = sess.run(self.top5_predictions, feed_dict={
                    self.x: imgs,
                    self.dropout_placeholder_conv: 1.0,
                    self.dropout_placeholder_fc: 1.0
                })
            else:
                preds = sess.run(self.prediction, feed_dict={
                    self.x: imgs,
                    self.dropout_placeholder_conv: 1.0,
                    self.dropout_placeholder_fc: 1.0
                })


        return preds

    def show_conv_feature_maps(self, img, conv_layer_idx=0, activation_min=-1, activation_max=-1,
                               plt_num=1, fig_size=(15, 15), title_y_pos=1.0):
        """
        Shows the resulting feature maps at a given convolutional level for a SINGLE image
        """
        # s = tf.train.Saver()
        with tf.Session(graph=self.graph) as sess:
            # Never forget to re-initialise the variables
            tf.global_variables_initializer()
            # tf.reset_default_graph()

            model_file_name = "{0}{1}.chkpt".format("lenet/", self.model_config.name)
            self.saver.restore(sess, model_file_name)

            # Run a prediction
            preds = sess.run(self.prediction, feed_dict={
                self.x: np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]]),
                self.dropout_placeholder_conv: 1.0,
                self.dropout_placeholder_fc: 1.0
            })

            var_name = "{0}/conv_{1}_relu:0".format(self.model_config.name, conv_layer_idx)
            print("Fetching tensor: {0}".format(var_name))
            conv_layer = tf.get_default_graph().get_tensor_by_name(var_name)

            activation = sess.run(conv_layer, feed_dict={
                self.x: np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]]),
                self.dropout_placeholder_conv: 1.0,
                self.dropout_placeholder_fc: 1.0
            })
            featuremaps = activation.shape[-1]
            # (1, 13, 13, 64)
            print("Shape of activation layer: {0}".format(activation.shape))

            # fix the number of columns
            cols = 8
            rows = featuremaps // cols
            fig, axes = plt.subplots(rows, cols, figsize=fig_size)
            k = 0
            for i in range(0, rows):
                for j in range(0, cols):
                    ax = axes[i, j]
                    featuremap = k

                    if activation_min != -1 & activation_max != -1:
                        ax.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                                  vmax=activation_max, cmap="gray")
                    elif activation_max != -1:
                        ax.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max,
                                  cmap="gray")
                    elif activation_min != -1:
                        ax.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                                  cmap="gray")
                    else:
                        ax.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")

                    ax.axis("off")
                    k += 1

            fig.suptitle("Feature Maps at layer: {0}".format(conv_layer), fontsize=12, fontweight='bold', y=title_y_pos)
            fig.tight_layout()
            plt.show()




def EdLeNet(x, mc, dropout_conv_pct, dropout_fc_pct):
    mu = 0
    sigma = 0.1

    prev_conv_layer = x
    conv_depth = mc.conv_depth_start
    conv_input_depth = mc.input_img_dimensions[-1]
    print("[EdLeNet] Building neural network [conv layers={0}, "
          "conv filter size={1}, conv start depth={2},"
          " fc layers={3}]".format(mc.conv_layers_count, mc.conv_filter_size, conv_depth, len(mc.fc_output_dims)))


    for i in range (0,mc.conv_layers_count):
        conv_output_depth = conv_depth*(2**(i))


        conv_W = tf.Variable(
            tf.truncated_normal(shape=(mc.conv_filter_size, mc.conv_filter_size, conv_input_depth, conv_output_depth),
                                mean=mu, stddev=sigma))
        print("THe weight of {0} conv layer is : {1}".format(i, conv_W))
        conv_b = tf.Variable(tf.zeros(conv_output_depth))

        conv_output = tf.nn.conv2d(prev_conv_layer, conv_W, strides=[1, 1, 1, 1], padding='VALID',
                                   name="conv_{0}".format(i)) + conv_b
        conv_output = tf.nn.relu(conv_output, name="conv_{0}_relu".format(i))

        conv_output = tf.nn.max_pool(conv_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv_output = tf.nn.dropout(conv_output, dropout_conv_pct)

        prev_conv_layer = conv_output
        conv_input_depth = conv_output_depth

    fc0 = flatten(prev_conv_layer)
    prev_layer = fc0

    #fully connected layer
    for output_dim in mc.fc_output_dims:
        fcn_W = tf.Variable(tf.truncated_normal(shape=(prev_layer.get_shape().as_list()[-1], output_dim),
                                                mean=mu, stddev=sigma))
        fcn_b = tf.Variable(tf.zeros(output_dim))

        prev_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(prev_layer, fcn_W) + fcn_b), dropout_fc_pct)

        # Final layer (Fully Connected)
    fc_final_W = tf.Variable(tf.truncated_normal(shape=(prev_layer.get_shape().as_list()[-1], mc.output_classes),
                                                 mean=mu, stddev=sigma))
    fc_final_b = tf.Variable(tf.zeros(mc.output_classes))
    logits = tf.matmul(prev_layer, fc_final_W) + fc_final_b

    return logits



mc_5x5 = ModelConfig(EdLeNet, "EdLeNet_Color_Norm_5x5_Dropout_0.40", [32, 32, 3], [5, 32, 2], [120, 84], n_classes, [0.75, 0.5])
print(mc_5x5.name)
me_c_norm_drpt_0_50_5x5 = ModelExecutor(mc_5x5, learning_rate=0.001)
(c_norm_drpt_0_50_5x5_tr_metrics, c_norm_drpt_0_50_5x5_val_metrics, c_norm_drpt_0_50_5x5_duration) = me_c_norm_drpt_0_50_5x5.train_model(X_train_normalised, y_train, X_valid_normalised, y_valid, epochs=100)
(c_norm_drpt_0_50_5x5_ts_accuracy, c_norm_drpt_0_50_5x5_ts_loss, c_norm_drpt_0_50_5x5_ts_duration) =  me_c_norm_drpt_0_50_5x5.test_model(X_test_normalised, y_test)






# plot model

def plot_model_results(metrics, axes, lbs, xlb, ylb, titles, fig_title, fig_size=(7, 5), epochs_interval=10):
    """
    Nifty utility function to plot results of the execution of our model
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(axes), figsize=fig_size)
    print("Length of axis: {0}".format(axs.shape))

    total_epochs = metrics[0].shape[0]
    x_values = np.linspace(1, total_epochs, num=total_epochs, dtype=np.int32)

    for m, l in zip(metrics, lbs):
        for i in range(0, len(axes)):
            ax = axs[i]
            axis = axes[i]
            ax.plot(x_values, m[:, axis], linewidth=2, label=l)
            ax.set(xlabel=xlb[i], ylabel=ylb[i], title=titles[i])
            ax.xaxis.set_ticks(np.linspace(1, total_epochs, num=int(total_epochs / epochs_interval), dtype=np.int32))
            ax.legend(loc='center right')

    plt.suptitle(fig_title, fontsize=14, fontweight='bold')
    plt.show()

metrics_arr = [c_norm_drpt_0_50_5x5_tr_metrics, c_norm_drpt_0_50_5x5_val_metrics]
lbs = ["5x5 training", "5x5 validation"]
plot_model_results(metrics_arr, [2, 1], lbs, ["Epochs", "Epochs"], ["Accuracy", "Loss"],
                   ["Accuracy vs Epochs",
                    "Loss vs Epochs"],
                   "Color (Normalised)  - Accuracy and Loss of models"
                    ,fig_size=(17, 5))
