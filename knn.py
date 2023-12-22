# https://pyimagesearch.com/2021/04/17/your-first-image-classifier-using-k-nn-to-classify-images/?_ga=2.245512013.851376902.1703040146-1842902230.1698424416
# The k-NN algorithm used here ONLY classifies images based solely on the pixel intensities that appear in an image, nothing else
from sklearn.neighbors import KNeighborsClassifier # implementation of the k-NN algorithm
from sklearn.preprocessing import LabelEncoder # converts labels represented as strings to integers
from sklearn.model_selection import train_test_split # convenience function for train/test set splitting
from sklearn.metrics import classification_report # evaluates the performance of the classifier
from pyimagesearch.preprocessing.simple_preprocessor import SimplePreprocessor
from pyimagesearch.datasets.simple_dataset_loader import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,  help="# of neighbors for classification (default is 1)")
ap.add_argument("-j", "--jobs", type=int, default=-1,  help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

RESIZE_DIM = 52
# initialize the image processor
sp = SimplePreprocessor(RESIZE_DIM, RESIZE_DIM)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], data.shape[1]*data.shape[2]*data.shape[3]))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits:
# 75% training
# 25% testing
# X refers to the data, Y refers to the corresponding labels
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))