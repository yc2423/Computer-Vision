import numpy as np
import skimage.transform
from keras.layers import Dense,GlobalAveragePooling2D,Dropout

cifar_classes = ['airplane', 'automobile', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
av_classes = ['animal', 'vehicle']

def labels_to_one_hot(labels, num_classes):
    '''
    Convert a list of class labels, drawn from a total of num_classes
    classes, to a list of one-hot vectors representing the labels.
    The value of each class label will range from 0 to num_classes - 1.

    Parameters:
        labels - (n x 1) list of labels
        num_classes - the total number of classes in the dataset

    Return:
        one_hot - (n x num_classes) list of one-hot vectors
    '''
    ### TODO-1 BEGINS HERE ###
    #raise NotImplementedError
    n,d = labels.shape
    ind = labels.flatten()
    one_hot = np.zeros((n,num_classes))
    one_hot[range(n),ind] = 1
    return one_hot

    ### TODO-1 ENDS HERE ###

def split_dataset(x, y, x_t, y_t, train_size, val_size, test_size):
    '''
    Split a subset of our training data (x, y) into training and validation datasets
    of size train_size and val_size. For simplicity, pick the first train_size examples
    from (x, y) to form your final training dataset, and use the next val_size
    examples to form your validation dataset. Sample the first test_size
    examples from the test data (x_t, y_t) to form your test dataset. Don't shuffle
    the dataset - CIFAR-10 comes shuffled for us!

    train_size + val_size is guaranteed to be lesser than or equal to the number of
    examples in x. test_size is guaranteed to be lesser than or equal to the number
    of examples in x_t.

    Parameters:
        x - the images of the complete training dataset
        y - the one-hot encoded labels of the complete training dataset
        x_t - the images of the complete test dataset
        y_t - the one-hot encoded labels of the complete test dataset
        train_size - number of samples to be drawn into the final training dataset
        val_size - number of samples to be drawn into the final validation dataset
        test_size - number of samples to be drawn into the final test dataset

    Returns:
        x_train - (train_size x image_size x image_size x 3) array of images
        y_train - (train_size x num_classes) array of one-hot vectors
        x_val - (val_size x image_size x image_size x 3) array of images
        y_val - (val_size x num_classes) array of one-hot vectors
        x_test - (test_size x image_size x image_size x 3) array of images
        y_test - (test_size x num_classes) array of one-hot vectors
    '''
    ### TODO-2a BEGINS HERE ###
    #raise NotImplementedError
    ind = 0
    x_train = x[ind:ind+train_size,:,:,:]
    y_train = y[ind:ind+train_size,:]
    ind+=train_size
    x_val = x[ind:ind+val_size,:,:,:]
    y_val = y[ind:ind+val_size,:]
    ind = 0
    x_test = x_t[ind:ind+test_size,:,:,:]
    y_test = y_t[ind:ind+test_size,:]

    return x_train,y_train,x_val,y_val,x_test,y_test


    ### TODO-2a ENDS HERE ###

def preprocess_dataset(x, image_size):
    '''
    Preprocess a dataset of images by resizing and normalizing the images as follows:
    1. Resize every image to size (image_size x image_size x 3)
    2. Normalize each image by subtracting the channel-wise mean of the given dataset,
       and dividing by the channel-wise standard deviation of the dataset.

    Use skimage.transform.resize to resize the images - we have already imported
    the skimage.transform module for you!

    Note that in a real scenario, the channel-wise mean and standard deviations would
    be computed across the whole dataset. However, for simplicity, here we will only
    compute it for the dataset of images, x, passed into the function.

    Parameters:
        x: the dataset of images of size (n x h x w x 3) to be preprocessed
        image_size: the square image_size to which each image should be resized

    Returns:
        x_p: the preprocessed dataset of images, of size
            (n x image_size x image_size x 3)
    '''
    ### TODO-2b BEGINS HERE ###
    #raise NotImplementedError
    n,h,w,c = x.shape
    x_p = np.zeros((n,image_size,image_size,3))
    for i in range(n):
        img = skimage.transform.resize(x[i,:,:,:],(image_size,image_size,3))
        for j in range(3):
            img[:,:,j]-=np.mean(img[:,:,j])
            img[:,:,j]/=np.std(img[:,:,j])
        x_p[i,:,:,:] = img
    return x_p
    ### TODO-2b ENDS HERE ###

def get_N_cifar_images(N, L, images, labels):
    '''
    Retrieves the first N images of label L from the given set
    of images and one-hot labels, as well as the string class
    of the given label L. Refer to the cifar_classes list at the
    top of the file for the class description of each CIFAR-10 label.

    Parameters:
        N: the number of images to retrieve (guaranteed to be within range)
        L: the label being queried for (guaranteed to be between 0 and 9,
            inclusive)
        images: the images contained in the dataset being queried
        labels: the one-hot labels corresponding to the labels

    Returns:
        class_string: the name of the class corresponding to label L; use
            the cifar_classes list for this
        query_images: a batch of images containing the result of the query;
            should have shape (N x h x w x c) where h, w, and c are the
            height, width, and channels of images in the given dataset
    '''
    ind = np.argwhere(labels[:,L] == 1)
    ind = ind.flatten()
    ind = ind[0:N]
    ### TODO-3 BEGINS HERE ###
    #raise NotImplementedError
    class_string = cifar_classes[L]
    query_images = images[ind,:,:,:]
    return class_string,query_images
    ### TODO-3 ENDS HERE ###

########## PART 1: QUESTIONS ##########
# Enter your written answers in the space provided below!
#
# 1. Why is it important to have train, validation, and test splits?
# Answer: If we only have one data set, we don't know the real performance of our classifier. 
# Validation can help us tune parameters, and testing can help us approximate the ture generalization error.
# 2. What was the original size of the CIFAR images before we resized them?
# Answer:
# 32 x 32 x 3
#######################################

def build_cifar_top(base_output):
    '''
    Adds layers to the top of the base_output tensor according
    to the given specification for CIFAR-10 classification.
    The function must add the following in order:
    - Global Average Pooling
    - Fully-connected layer with 256 units (ReLU activation)
    - Dropout layer which randomly drops inputs at a rate of 50%
    - Softmax layer that produces 10 outputs

    As an example, we have added the Global Average Pooling layer for
    you. Reference the Keras documentation for common layers at
    https://keras.io/layers/core/, and feel free to import any
    necessary layers at the top of this file.

    Parameters:
        base_output: the tensor that represents the output of the base model

    Returns:
        cifar_output: the tensor that builds the specified layers on
            top of base_output
    '''
    cifar_output = GlobalAveragePooling2D()(base_output)

    ### TODO-4 BEGINS HERE ###
    #raise NotImplementedError
    cifar_output = Dense(256, activation='relu')(cifar_output)
    cifar_output = Dropout(0.5)(cifar_output)
    cifar_output = Dense(10, activation='softmax')(cifar_output)

    ### TODO-4 ENDS HERE ###
    return cifar_output

def freeze_model_weights(model, to_freeze):
    '''
    Freeze the weights of the first to_freeze layers of the given
    model.

    Hint: You can access the layers of a model as a list using
        model.layers. A layer can be frozen by setting its
        'trainable' property to False.

    Parameters:
        model: the network whose layers are to be frozen
        to_freeze: the number of layers to be frozen, starting from
            the beginning of the network (to_freeze = 0 should freeze
            no layers, to_freeze = 1 should freeze the first layer, etc.)

    Returns:
        None (update the model's layers in place)
    '''
    ### TODO-5 BEGINS HERE ###
    #raise NotImplementedError
    for layer in model.layers[0:to_freeze]:
        layer.trainable = False
    for layer in model.layers[to_freeze:]:
        layer.trainable = True
    ### TODO-5 ENDS HERE ###

def generate_predictions(model, image_batch):
    '''
    Generate class predictions for each image in the given
    image_batch using the provided model. You might find
    Keras' model.predict function useful
    (https://keras.io/models/model/).

    Parameters:
        model: the model to be used to generate the predictions
        image_batch: (n x h x w x c) batch of images to be
            predicted, where n is the number of images, each of
            dimension h x w x c.

    Returns:
        labels: (n x 1) list of integer labels of
            the classes predicted for each of the n examples
            in the batch
        scores: (n x 1) list of scores containing the
            scores of the predictions for each of the n examples
            in the batch
    '''
    ### TODO-6 BEGINS HERE ###
    #raise NotImplementedError
    distribution = model.predict_on_batch(image_batch)
    labels = np.argmax(distribution,axis = 1)
    scores = np.max(distribution,axis = 1)
    return labels,scores
    ### TODO-6 ENDS HERE ###

########## PART 2: QUESTIONS ##########
# Enter your written answers in the space provided below!
#
# 1. What is an 'epoch'?
# Answer: one epoch is one forward progress and one backward progress for all training data.
#
# 2. What is meant by 'batch size'?
# Answer: The number of training images for one forward progress.
#
# 3. How does the loss function being used relate to negative log likelihood
#    discussed in class?
# Answer: The loss function is equal to negtive log likelihood.
#
# 4. List the final training, validation, and test accuracies of your model.
# Answer:
#    training: 0.6258 / validation : 0.602 / testing : 0.574 
#
# 5. Suggest one change to the layers that we added that could result in
#    a higher accuracy.
# Answer: Lowering the dropout rate may help.
#
#######################################

def change_labels_av(y_train, y_val, y_test):
    '''
    Uses the original CIFAR-10 one-hot labels to create
    new labels for the animals vs vehicles classification
    problem. Animals should correspond to class 0, and
    vehicles should correspond to class 1. Refer to the
    cifar_classes list at the top of the file for the class
    description of each CIFAR-10 label. 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse' should go to the animal
    class, and the other labels should go to the vehicles
    class.

    Parameters:
        y_train: one-hot encoded training labels from CIFAR-10
            of size (p x 10), p = number of training examples
        y_val: one-hot encoded validation labels from CIFAR-10
            of size (q x 10), q = number of validation examples
        y_test: one-hot encoded test labels from CIFAR-10
            of size (r x 10), r = number of test examples

    Returns:
        y_train_av: one-hot encoded training labels for the
            animals vs vehicles problem (p x 2)
        y_val_av: one-hot encoded validation labels for the
            animals vs vehicles problem (q x 2)
        y_test_av: one-hot encoded test labels for the animals
            vs vehicles problem (r x 2)
    '''
    ### TODO-7 BEGINS HERE ###
    #raise NotImplementedError
    #cifar_classes = ['airplane', 'automobile', 'bird', 'cat',
    #    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    #av_classes = ['animal', 'vehicle']


    y_train_av = np.zeros((len(y_train),2))
    
    a = np.sum(y_train[:,0:2],axis = 1)
    y_train_av[a == 1,1] = 1
    b = np.sum(y_train[:,2:8],axis = 1)
    y_train_av[b == 1,0] = 1
    c = np.sum(y_train[:,8:],axis = 1)
    y_train_av[c == 1,1] = 1

    print (np.sum(y_train_av[:,0]),np.sum(y_train_av[:,1]))
    y_val_av = np.zeros((len(y_val),2))
    
    a = np.sum(y_val[:,0:2],axis = 1)
    y_val_av[a == 1,1] = 1
    b = np.sum(y_val[:,2:8],axis = 1)
    y_val_av[b == 1,0] = 1
    c = np.sum(y_val[:,8:],axis = 1)
    y_val_av[c == 1,1] = 1

    y_test_av = np.zeros((len(y_test),2))

    a = np.sum(y_test[:,0:2],axis = 1)
    y_test_av[a == 1,1] = 1
    b = np.sum(y_test[:,2:8],axis = 1)
    y_test_av[b == 1,0] = 1
    c = np.sum(y_test[:,8:],axis = 1)
    y_test_av[c == 1,1] = 1

    return y_train_av,y_val_av,y_test_av
    ### TODO-7 ENDS HERE ###

def build_av_top(cifar_output):
    '''
    Adds a 2-unit softmax layer to the given cifar_output base tensor.
    As before, refer to the Keras layers documentation, and import any
    necessary Keras layers at the top of the file.

    Parameters:
        cifar_output: the tensor that represents the output of the
            CIFAR model

    Returns:
        av_output: the tensor that builds the specified layer onto the
            top of cifar_output
    '''
    ### TODO-8 STARTS HERE ###
    #raise NotImplementedError
    av_output = Dense(2, activation='softmax')(cifar_output)
    return av_output
    ### TODO-8 ENDS HERE ###

########## PART 3: QUESTIONS ##########
# Enter your written answers in the space provided below!
#
# 1. Approximately what percentage of our training examples
#    belong to the animal class?
# Answer: Becuase our model is very accurate for classifying both vehicles and animals,
# the percentage difference between the two classes should be small. However, the batch socre of correct classification 
# of animal images is often higher than the batch score of correct classification of vehicle images, so we guess the percentage of animal
# examples should be around 55% to 60%.
# 2. What would happen if less than 10% of our training examples
#    belonged to the animal class?
# Answer: If a given image contain vehicles , our model can give it correct label with very high batch score. However, our model may also recognize animals
# as vehicles.
# 3. How many trainable parameters does av_model contain?
#    Briefly explain how you arrived at this number.
# Answer: 1 parameter.
# Only the softmax layer is trainable, so we only have to consider the number of parameters in softmax function.
# Becuase we only have 2 classes, the the softmax function is almost equal to logistic regression. If we don't consider the bias, there is only 
# one parameter can be trained.
#######################################

def generate_occlusions(image, occlusion_size):
    '''
    This function slides an occlusion window of size
    (occlusion_size x occlusion_size) in a non-overlapping way over the rows
    and columns of the image. Pixels within the occlusion window should all
    be set to 0, so that the network cannot retrieve any information about
    that region. It is guranteed that the image height and width will be
    exactly divisible by the given occlusion_size!

    Please generate images in a *row-major order*, going from left-to-right
    within each row, and going from top-to-bottom through the rows. For
    example, if our images were 2x2 and the occlusion_size was 1 pixel, we
    would return occluded images in the following order:

    1) x .   2) . x   3) . .   4) . .
       . .      . .      x .      . x

    where x represents an occlusion and . represents an unoccluded pixel.

    Parameters:
        image: the input image of size (h x w x c) to be occluded
        occlusion_size: the size of the square occlusion window

    Returns:
        occ_batch: a numpy array of size
            (((h / occlusion_size) * (w / occlusion_size)) x h x w x c). This is
            a batch of occluded images, each of which is an occlusion of the
            original image in a different position. The order of generated images
            matters, see the above specification!
    '''
    ### TODO-9 BEGINS HERE ###
    #raise NotImplementedError
    h,w,c = image.shape
    occ_batch = np.zeros((((h / occlusion_size) * (w / occlusion_size)),h,w,c))
    for i in range(0,(h / occlusion_size)):
        for j in range(0,(w / occlusion_size)):
            new_image = image.copy()
            new_image[i*occlusion_size:(i+1)*occlusion_size,j*occlusion_size:(j+1)*occlusion_size,:] = 0
            occ_batch[i*(w / occlusion_size)+j,:,:,:] = new_image.copy()
    return occ_batch
    ### TODO-9 BEGINS HERE ###

def find_worst_occlusion(images, model, y):
    '''
    Given a batch of occluded images, this function finds the occluded image
    that contains the occlusion to which the given model is most sensitive. To
    do so, the function finds the model's prediction for each image in the
    batch. It then picks the image that produced the smallest score for the
    correct class label.

    Parameters:
        images: (n x h x w x c) batch of images, each of which contains an
            occlusion of some original base image
        model: the model whose occlusion sensitivity is being evaluated
        y: the one-hot encoded label corresponding to the correct class of the
            original base image

    Returns:
        worst_occ_im: the image of size (h x w x c) from images for which the
            model produced the lowest score on the correct class label
        worst_occ_score: the score generated by the model for worst_occ_im on
            the correct class label
    '''
    ### TODO-10 BEGINS HERE ###
    #raise NotImplementedError
    label = np.argmax(y)
    distribution = model.predict_on_batch(images)
    index = np.argmin(distribution[:,label])
    worst_occ_im = images[index]
    worst_occ_score = distribution[index,label]

    return worst_occ_im,worst_occ_score
    ### TODO-10 ENDS HERE ###

def generate_adversarial(image, get_gradients, alpha, num_steps):
    '''
    Perform gradient ascent on the given image for num_steps iterations with a
    step size of alpha. The gradient of the score of the target class can be
    retrieved by calling the function get_gradients with an input image. The
    final adversarial image is returned, along with the difference between the
    adversarial and the original image. This difference needs to be normalized
    per-pixel!

    Parameters:
        image: (h x w x c) image which we are transforming into an adversarial
            example
        get_gradients: a function that accepts a batch of images, and returns
            the score of the target class, and its gradient with respect to
            each image in the batch
        alpha: the step size to be used while performing gradient ascent
        num_steps: the number of gradient ascent steps/iterations to execute

    Returns:
        changed_image: (h x w x c) image that represents the final
            adversarial example obtained after gradient ascent
        diff_image: (h x w) image representing the per-pixel normalized
            difference between the changed_image and the original image.
    '''
    ### TODO-11 BEGINS HERE ###
    #raise NotImplementedError
    changed_image = image.copy()
    for i in range(num_steps):
        target_score, grads = get_gradients([np.array([changed_image,changed_image])])
        changed_image+= alpha*grads[0]

    diff_image =  np.sqrt(np.sum((image - changed_image)**2,axis = 2))

    return changed_image,diff_image

    ### TODO-11 ENDS HERE ###
