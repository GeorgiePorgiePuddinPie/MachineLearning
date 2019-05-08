# First, we are going to load the file names and their respective target labels into numpy array! 
from sklearn.datasets import load_files
import numpy as np

train_dir = '/Users/georgehardy/Desktop/fruits-360/Training'
test_dir = '/Users/georgehardy/Desktop/fruits-360/Test'

def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files,targets,target_labels
    
x_train, y_train,target_labels = load_dataset(train_dir)
x_test, y_test,_ = load_dataset(test_dir)

# Let's confirm the number of classes :p
no_of_classes = len(np.unique(y_train))
no_of_classes

print(y_train[0:10])
# target labels are numbers corresponding to class label. We need to change them to a vector of 81 elements.


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train,no_of_classes)
y_test = np_utils.to_categorical(y_test,no_of_classes)
y_train[0]
 # Note that only one element has value 1(corresponding to its label) and others are 0.

x_test,x_valid = x_test[7000:],x_test[:7000]
y_test,y_vaild = y_test[7000:],y_test[:7000]

# training data is just file names of images. We need to convert them into pixel matrix.

# We just have the file names in the x set. Let's load the images and convert them into array.
from keras.preprocessing.image import array_to_img, img_to_array, load_img

def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array

x_train = np.array(convert_image_to_array(x_train))

x_valid = np.array(convert_image_to_array(x_valid))

x_test = np.array(convert_image_to_array(x_test))


# there are elements will other values too :p
# time to re-scale so that all the pixel values lie within 0 to 1
x_train = x_train.astype('float32')/255
x_valid = x_valid.astype('float32')/255
x_test = x_test.astype('float32')/255

from sklearn import svm

model = svm.SVC(kernel='linear', C=1, gamma=1) 

model.fit(x_train, y_train)

model.score(x_train, y_train)

#Predict Output

predicted= model.predict(x_test)

