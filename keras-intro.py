import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers

# from keras.datasets import mnist     # MNIST dataset is included in Keras%
# from keras.models import Sequential  # Model type to be used

# from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
# from keras.utils import np_utils                         # NumPy related tools

# from keras import optimizers


from sklearn.metrics import confusion_matrix
import itertools
# from tensorflow import keras
# import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ######################################
# # The MNIST data is split between 60,000 28 x 28 pixel training images and 10,000 28 x 28 pixel images
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# print("X_train shape", X_train.shape)
# print("y_train shape", y_train.shape)
# print("X_test shape", X_test.shape)
# print("y_test shape", y_test.shape)

# #######################################
# # Using matplotlib, we can plot some sample images from the training set directly into this Jupyter Notebook. We can egzamine the interclass variability - how many different ways of writing the same digit there are!
# plt.rcParams['figure.figsize'] = (9,9) # Make the figures a bit bigger

# def visualize_classes(X, y):
#   for i in range(0, 10):
#     img_batch = X[y == i][0:10]
#     img_batch = np.reshape(img_batch, (img_batch.shape[0]*img_batch.shape[1], img_batch.shape[2]))
#     if i > 0:
#       img = np.concatenate([img, img_batch], axis = 1)
#     else:
#       img = img_batch
#   plt.figure(figsize=(10,20))
#   plt.axis('off')
#   plt.imshow(img, cmap='gray')


# visualize_classes(X_train, y_train)
# # plt.show()

# # just a little function for pretty printing a matrix
# def matprint(mat, fmt="g"):
#     col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
#     for x in mat:
#         for i, y in enumerate(x):
#             print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
#         print("")

# # now print!        
# # matprint(X_train[4])

# from sklearn.manifold import TSNE
# import matplotlib.patheffects as PathEffects
# import seaborn as sns

# RS=19238

# class_names = [ str(clid) for clid in range(10) ]
    
# def scatter(x, colors):
#     # We choose a color palette with seaborn.
#     palette = np.array(sns.color_palette("hls", 10))

#     # We create a scatter plot.
#     f = plt.figure(figsize=(8, 8))
#     ax = plt.subplot(aspect='equal')
#     sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
#                     c=palette[colors.astype(np.int)])
#     plt.xlim(-25, 25)
#     plt.ylim(-25, 25)
#     ax.axis('off')
#     ax.axis('tight')

#     # We add the labels for each digit.
#     txts = []
#     for i in range(10):
#         # Position of each label.
#         xtext, ytext = np.median(x[colors == i, :], axis=0)
#         txt = ax.text(xtext, ytext, class_names[i], fontsize=15)
#         txt.set_path_effects([
#             PathEffects.Stroke(linewidth=5, foreground="w"),
#             PathEffects.Normal()])
#         txts.append(txt)
        
# def plot_tsne(X, y):
#   print('calculating tsne ...')
#   proj = TSNE(random_state=RS).fit_transform(X)
#   scatter(proj, y)


# X = np.reshape(X_train, (X_train.shape[0], 28 * 28))[0:2000]
# y = y_train[0:2000]
# plot_tsne(X, y)
# plt.show()

#########################################################
#########################################################

# import some additional tools

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization

from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')         # change integers to 32-bit floating point numbers
X_test = X_test.astype('float32')

X_train /= 255                              # normalize each value for each pixel for the entire vector for each input
X_test /= 255

y_train = y_train.reshape((1,-1))[0]
y_test = y_test.reshape((1,-1))[0]

print("Training matrix shape", X_train.shape, y_train.shape)
print("Testing matrix shape", X_test.shape, y_test.shape)

# one-hot format classes

nb_classes = 10 # number of unique digits

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

cifar_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(0, 10):
  img_batch = X_train[y_train == i][0:10]
  img_batch = np.reshape(img_batch, (img_batch.shape[0]*img_batch.shape[1], img_batch.shape[2], img_batch.shape[3]))
  if i > 0:
    img = np.concatenate([img, img_batch], axis = 1)
  else:
    img = img_batch
plt.figure(figsize=(10,20))
plt.axis('off')
plt.imshow(img, cmap='gray')

####################################################################
model = Sequential()                                 # Linear stacking of layers

# Convolution Layer 1
model.add(Conv2D(512, (3, 3), input_shape=(32,32,3)))
model.add(Activation('relu') )                       # activation
model.add(MaxPooling2D(pool_size=(2,2)))             # Pool the max values over a 2x2 kernel

# Convolution Layer 2
model.add(Conv2D(32, (3, 3)))                        # 32 different 5x5 kernels -- so 32 feature maps
model.add(Activation('relu'))                        # activation
model.add(MaxPooling2D(pool_size=(2,2)))             # Pool the max values over a 2x2 kernel

model.add(Flatten())                                 # Flatten final output matrix into a vector

# Fully Connected Layer 
model.add(Dense(1024))                                # 128 FC nodes
model.add(Activation('relu'))                        # activation
model.add(Dropout(0.2))

# Fully Connected Layer 
model.add(Dense(128))                                # 128 FC nodes
model.add(Activation('relu'))                        # activation

# Fully Connected Layer                        
model.add(Dense(10))                                 # final 10 FC nodes
model.add(Activation('softmax'))                     # softmax activation

model.summary()


adam = optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08, validation_split=0.2)

train_generator = gen.flow(X_train, Y_train, batch_size=128, subset='training')
valid_generator = gen.flow(X_train, Y_train, batch_size=128, subset='validation')



model.fit_generator(train_generator, steps_per_epoch=50000//128, epochs=50, verbose=1, validation_data=valid_generator, validation_steps = 10000 // 128)


######################################################################
# Test

score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = model.predict_classes(X_test)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]

incorrect_indices = np.nonzero(predicted_classes != y_test)[0]


cnf_matrix = confusion_matrix(y_test, predicted_classes)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=cifar_names,
                      title='Confusion matrix, without normalization')


def show_samples_rgb(indices, preds, images, labels, count=3, names = []):
    plt.figure()
    for i, sample in enumerate(indices[:count**2]):
        plt.subplot(count,count,i+1)
        plt.imshow(images[sample], interpolation='none')
        plt.axis('off')
        if len(names) > 0:
          plt.title("Predicted {}\nClass {}".format(names[int(preds[sample])], names[int(labels[sample])]))
        else:
          plt.title("Predicted {}\nClass {}".format(preds[sample], labels[sample]))          
    
    plt.tight_layout()

# Poprawne klasyfikacje
show_samples_rgb(correct_indices, predicted_classes, X_test, y_test, 5, cifar_names)

# Błędne klasyfikacje
show_samples_rgb(incorrect_indices, predicted_classes, X_test, y_test, 5, cifar_names)

# plt.show()
# input("Press Enter to continue...")