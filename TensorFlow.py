import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def get_dataset():
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()
    training_images = training_images/255.0
    test_images = test_images/255.0
    return training_images, training_labels, test_images, test_labels

def training(training_images, training_labels, test_images):
        
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy')
    model.fit(training_images, training_labels, epochs=5)
    
    return model.predict(test_images)

def predicted_low_acurracy(classifications, accuracy_limit = 0.65):
    predicted_low_acurracy_indexs = []
    for i in range(len(classifications)):
        accuracy = classifications[i][np.argmax(classifications[i])]
        if accuracy < accuracy_limit: 
            predicted_low_acurracy_indexs.append(i)
    return predicted_low_acurracy_indexs

def plotting_low_accuracy_images(classifications, test_images, predicted_low_acurracy_indexs):
    fig = plt.figure()
    lenght = len(predicted_low_acurracy_indexs)
    colums = int(math.sqrt(lenght))
    row = int(math.sqrt(lenght)) + 1
    
    for cnt, index in enumerate(predicted_low_acurracy_indexs):
        porcentage = max(classifications[index])
        detected_num = np.argmax(classifications[index])
        print(classifications[index])

        plot = fig.add_subplot(row, colums, cnt)
        plot.set_title("{} : {:.3f}".format(detected_num, porcentage))
        plot.imshow(test_images[index])
        plot.axis("off")
    plt.show()

