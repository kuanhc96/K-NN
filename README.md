# K-NN
This is a simple implementation of image classification using the k-NN algorithm from sci-kit learn.
By examining ONLY the pixel intensities of input images, a feature vector can be formed, and classification performed based on these feature vectors.
Process:
1. load an image
2. resize this image to some standard size (having a standard size is more important than resizing itself; resizing just makes sure the feature vectors aren't too big)
3. fit the k-NN algorithm using the input training set images
4. test set images can be classified based on its distances from other classified images

This method is only slightly better than randomly guessing, and results are highly dependent on the set of pixel intensities that appear in each class.
For example, the classiier can separate pandas from dogs and cats decently well, since pandas have a pretty standard look and colors.
However, cats and dogs, given their diversity in looks and colors, cannot be separated very well from each other.
This algorithm is easy to understand, but does not perform very well in real-life situations.