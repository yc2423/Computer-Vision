# segment.py - Segments an input image.
# Cornell University CS 4670/5670: Intro Computer Vision
import math
import numpy as np
import scipy
import scipy.sparse
import scipy.spatial
from scipy.sparse import spdiags
from scipy.signal import convolve2d
from scipy.spatial import distance
from scipy import linalg
from scipy.linalg import eigh,fractional_matrix_power
import cv2
from colors import COLORS
 
#########################################################
###    Part A: Image Processing Functions
######################################################### 

# TODO:PA2 Fill in this function
def normalizeImage(cvImage, minIn, maxIn, minOut, maxOut):
    '''
    Take image and map its values linearly from [min_in, max_in]
    to [min_out, max_out]. Assume the image does not contain 
    values outside of the [min_in, max_in] range.
    
    Parameters:
        cvImage - a (m x n) or (m x n x 3) image.
        minIn - the minimum of the input range
        maxIn - the maximum of the input range
        minOut - the minimum of the output range
        maxOut - the maximum of the output range
        
    Return:
        renormalized - the linearly rescaled version of cvImage.
    '''
    renormalized = (cvImage - minIn)/(maxIn - minIn)*(maxOut - minOut) + minOut
    #print 'in normalizeImage function'
    return renormalized

# TODO:PA2 Fill in this function
def getDisplayGradient(gradientImage):
    """
    Use the normalizeImage function to map a 2d gradient with one
    or more channels such that where the gradient is zero, the image
    has 50% percent brightness. Brightness should be a linear function 
    of the input pixel value. You should not clamp, and 
    the output pixels should not fall outside of the range of the uint8 
    datatype.
    
    Parameters:
        gradientImage - a per-pixel or per-pixel-channel gradient array
                        either (m x n) or (m x n x 3). May have any 
                        numerical datatype.
    
    Return:
        displayGrad - a rescaled image array with a uint8 datatype.
    """
    minIn = np.float(np.min(gradientImage))
    maxIn = np.float(np.max(gradientImage))
    #print minIn,maxIn

    minOut = 0.0
    if minIn <= 0.01 and minIn >= - 0.01:
        minOut = 127.0
        maxOut = 255.0
    else:
        maxOut = maxIn/(-minIn)*127.0 + 127.0
        maxOut = np.min(maxOut,255.0)
    displayGrad = normalizeImage(gradientImage,minIn,maxIn,minOut,maxOut)
    displayGrad = np.array(displayGrad,dtype=np.uint8)
    return displayGrad
    #raise NotImplementedError

# TODO:PA2 Fill in this function
def takeXGradient(cvImage):
    '''
    Compute the x-derivative of the input image with an appropriate
    Sobel implementation. Should return an array made of floating 
    point numbers.
    

    Parameters:
        cvImage - an (m x n) or (m x n x 3) image
        
    Return:
        xGrad - the derivative of cvImage at each position w.r.t. the x axis.
    
    '''
    
    k_row = np.array([-1.0,0.0,1.0])
    k_col = np.array([1.0,2.0,1.0])

    fig_size = cvImage.shape
    height = fig_size[0]
    width = fig_size[1]
    row_img = np.zeros_like(cvImage)
    m,n = 3,3
    
    for i in range(0,height):
        for j in range(0,width):
            r_low = max(j-(n/2),0)
            r_up = min(j+(n+1)/2,width)
            if len(fig_size) == 2:
                row_img[i][j] = np.sum(cvImage[i,r_low:r_up]*k_row[r_low - j + n/2:r_up-j+n/2])
            else:
                for x in range(0,3):
                    row_img[i][j][x] = np.sum(cvImage[i,r_low:r_up,x]*k_row[r_low - j + n/2:r_up-j+n/2])

    xGrad = np.zeros_like(cvImage)
    
    for i in range(0,height):
        for j in range(0,width):
            c_low = max(i-(m/2),0)
            c_up = min(i+(m+1)/2,height)
            if len(fig_size) == 2:
                xGrad[i][j] = np.sum(row_img[c_low:c_up,j]*k_col[c_low-i+m/2:c_up - i + m/2])
            else:
                for x in range(0,3):
                    xGrad[i][j][x] = np.sum(row_img[c_low:c_up,j,x]*k_col[c_low-i+m/2:c_up - i + m/2])
    

    return xGrad
    '''
    img = cvImage
    fig_size = img.shape
    #img = img.astype(float)
    #kernel = kernel.astype(float)
    height = fig_size[0]
    width = fig_size[1]
    new_img = np.zeros_like(img)
    kernel = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
    m,n = kernel.shape
    for i in range(0,height):
        for j in range(0,width):
        #    for k in range(max(i-(m/2),0),min(i+(m+1)/2,height)):
        #        for l in range(max(j-(n/2),0),min(j+(n+1)/2,width)):
        #            new_img[i][j] = new_img[i][j] + img[k][l]*kernel[k-i+m/2][l-j+n/2]
            k_low = max(i-(m/2),0)
            k_up = min(i+(m+1)/2,height)
            l_low = max(j-(n/2),0)
            l_up = min(j+(n+1)/2,width)
        #    print k_low,k_up,l_low,l_up
        #    print k_low-i+m/2,k_up-i+m/2,l_low - j + n/2,l_up-j+n/2
            if len(fig_size) == 2:
                new_img[i][j] = np.sum(img[k_low:k_up,l_low:l_up]*kernel[k_low-i+m/2:k_up - i + m/2,l_low - j + n/2:l_up-j+n/2],axis=None)
            else:
                for x in range(0,3):
                    new_img[i][j][x] = np.sum(img[k_low:k_up,l_low:l_up,x]*kernel[k_low-i+m/2:k_up - i + m/2,l_low - j + n/2:l_up-j+n/2],axis=None)
    return new_img
    '''
    
# TODO:PA2 Fill in this function
def takeYGradient(cvImage):
    '''
    Compute the y-derivative of the input image with an appropriate
    Sobel implementation. Should return an array made of floating 
    point numbers.
    
    Parameters:
        cvImage - an (m x n) or (m x n x 3) image
        
    Return:
        yGrad - the derivative of cvImage at each position w.r.t. the y axis.
    '''
    k_row = np.array([1.0,2.0,1.0])
    k_col = np.array([-1.0,0.0,1.0])
    fig_size = cvImage.shape
    height = fig_size[0]
    width = fig_size[1]
    row_img = np.zeros_like(cvImage)
    m,n = 3,3
    
    for i in range(0,height):
        for j in range(0,width):
            r_low = max(j-(n/2),0)
            r_up = min(j+(n+1)/2,width)
            if len(fig_size) == 2:
                row_img[i][j] = np.float(np.sum(cvImage[i,r_low:r_up]*k_row[r_low - j + n/2:r_up-j+n/2]))
            else:
                for x in range(0,3):
                    row_img[i][j][x] = np.float(np.sum(cvImage[i,r_low:r_up,x]*k_row[r_low - j + n/2:r_up-j+n/2]))


    yGrad = np.zeros_like(cvImage)

    for i in range(0,height):
        for j in range(0,width):
            c_low = max(i-(m/2),0)
            c_up = min(i+(m+1)/2,height)
            if len(fig_size) == 2:
                yGrad[i][j] = np.sum(row_img[c_low:c_up,j]*k_col[c_low-i+m/2:c_up - i + m/2])
            else:
                for x in range(0,3):
                    yGrad[i][j][x] = np.sum(row_img[c_low:c_up,j,x]*k_col[c_low-i+m/2:c_up - i + m/2])

    #print np.min(yGrad),np.max(yGrad)
    return yGrad
    #raise NotImplementedError
    
# TODO:PA2 Fill in this function
def takeGradientMag(cvImage):
    '''
    Compute the gradient magnitude of the input image for each 
    pixel in the image. 
    
    Parameters:
        cvImage - an (m x n) or (m x n x 3) image
        
    Return:
        gradMag - the magnitude of the 2D gradient of cvImage. 
                  if multiple channels, handle each channel seperately.
    '''
    xGrad = takeXGradient(cvImage)
    yGrad = takeYGradient(cvImage)
    gradMag = np.zeros_like(cvImage)

    if len(gradMag.shape) == 3:
        for x in range(0,3):
            gradMag[:,:,x] = np.sqrt(xGrad[:,:,x]**2 + yGrad[:,:,x]**2)
    else:
        gradMag = np.sqrt(xGrad**2 + yGrad**2)

    return gradMag
    #raise NotImplementedError

#########################################################
###    Part B: k-Means Segmentation Functions
######################################################### 

# TODO:PA2 Fill in this function
def chooseRandomCenters(pixelList, k):
    """
    Choose k random starting point from a list of candidates.
    
    Parameters:
        pixelList - an (n x 6) matrix 
        
    Return:
        centers - a (k x 6) matrix composed of k random rows of pixelList
    """
    #raise NotImplementedError
    m,n = pixelList.shape
    ind = np.random.randint(m,size=k)
    centers = pixelList[ind,:]
    return centers

# TODO:PA2 Fill in this function
def kMeansSolver(pixelList, k, centers=None, eps=0.001, maxIterations=100):
    '''
    Find a local optimum for the k-Means problem in 3D with
    Lloyd's Algorithm
    
    Assign the index of the center closest to each pixel
    to the fourth element of the row corresponding that 
    pixel.
    
    Parameters:
        pixelList - n x 6 matrix, where each row is <H, S, V, x, y, c>,
                    and c is index the center closest to it.
        centers - a k x 5 matrix where each row is one of the centers
                  in the k means algorithm. Each row is of the format
                  <H, S, V, x, y>
        eps - a positive real number user to test for convergence.
        maxIterations - a positive integer indicating how many 
                        iterations may occur before aborting.
    
    Return:
        iter - the number of iterations before convergence.
    '''
    # TODO:PA2 
    # H,S,V, x, and y values into the [0, 1] range.
    # raise NotImplementedError

    
    mini = pixelList[:,0:5].min(axis=0)
    maxi = pixelList[:,0:5].max(axis=0)

    if centers != None:
        mini_c = centers.min(axis=0)
        maxi_c = centers.max(axis=0)

        temp_mini = np.array([mini,mini_c])
        temp_maxi = np.array([maxi,maxi_c])

        mini = temp_mini.min(axis=0)
        maxi = temp_maxi.max(axis=0)

    for i in range(0,5):
        if maxi[i] == mini[i]:
            maxi[i] = 1.0 + mini[i]

    pixelList[:,0:5] = (pixelList[:,0:5] - mini)/(maxi - mini)
    if centers != None:
        centers = (centers - mini)/(maxi - mini)



    # Initialize any data structures you need.
    #raise NotImplementedError
    n,m = pixelList.shape
    # END TODO:PA2
    
    if centers is None:
        centers = chooseRandomCenters(pixelList,k)[:,0:5]
   
    for iter in range(maxIterations):
        count = np.zeros(k)
        dist_sum = np.zeros((k,5))
        # TODO:PA2 Assign each point to the nearest center
        #raise NotImplementedError
        for i in range(0,n):
            dist = np.sqrt(np.sum((centers-pixelList[i,0:5])**2,axis = 1))
            ind = np.argmin(dist)
            pixelList[i,5] = ind
            count[ind] = count[ind]+1
            dist_sum[ind] = dist_sum[ind] + pixelList[i,0:5]
        # END TODO:PA2
        # TODO:PA2 Recalculate centers
        #raise NotImplementedError
        #print dist_sum
        #print count
        newCenters = np.transpose(dist_sum.transpose()/count) 
        #print newCenters

        # END TODO:PA2
        
        validCenters = np.isfinite(newCenters)
        if (np.linalg.norm(centers[validCenters] - newCenters[validCenters], 2) < eps):
            centers = newCenters
            return iter
        else:
            centers = newCenters
    return iter 
      
def convertToHsv(rgbTuples):
    """
    Convert a n x 3 matrix whose rows are RGB tuples into
    an n x 3 matrix whose rows are the corresponding HSV tuples.
    The entries of rgbTuples should lie in [0,1]
    """
    B = rgbTuples[:,0]
    G = rgbTuples[:,1]
    R = rgbTuples[:,2]
    
    hsvTuples = np.zeros_like(rgbTuples)
    H = hsvTuples[:,0]
    S = hsvTuples[:,1]
    V = hsvTuples[:,2]
    
    alpha = 0.5 * (2*R - G - B)
    beta = np.sqrt(3)/2 * (G - B)
    H = np.arctan2(alpha, beta)
    V = np.max(rgbTuples,1)
    
    chroma = np.sqrt(np.square(alpha) + np.square(beta))
    S[V != 0] = np.divide(chroma[V != 0], V[V != 0])
    
    hsvTuples[:,0] = H  
    hsvTuples[:,1] = S  
    hsvTuples[:,2] = V  
    
    return hsvTuples
            
    
def kMeansSegmentation(cvImage, k, useHsv=True, eps=1e-14):
    """
    Execute a color-based k-means segmentation 
    """
    # Reshape the imput into a list of R,G,B,X,Y,C tuples, where
    # means that a pixel has not yet been assigned to a segment.
    m, n = cvImage.shape[0:2]
    numPix = m*n
    pixelList = np.zeros((numPix,6))
    pixelList[:,0:3] = cvImage.reshape((numPix,3))
    pixelList[:,3] = np.tile(np.arange(n),m)
    pixelList[:,4] = np.repeat(np.arange(m), n)
    
    # Convert the image to hsv.
    if useHsv:
        pixelList[:,:3] = convertToHsv(pixelList[:,:3]/255.)*255
    
    # Initialize k random centers in the color-position space.
    centers = (np.max(pixelList[:,0:5],0)-np.min(pixelList[:,0:5],0))*np.random.random((k,5))+np.min(pixelList[:,0:5],0)

    # Run Lloyd's algorithm until convergence
    iter = kMeansSolver(pixelList, k, eps=eps)
    
    # Color the pixels based on their centers
    if k <= 64:
        colors = np.array(COLORS[:k])
    else:
        colors = np.random.uniform(0,255,(k,3))
    
    R = pixelList[:,0]
    G = pixelList[:,1]
    B = pixelList[:,2]
    centerIndices = pixelList[:,5]
    
    for j in range(k):
       R[centerIndices == j] = colors[j,0] 
       G[centerIndices == j] = colors[j,1]
       B[centerIndices == j] = colors[j,2]
       
    return pixelList[:,:3].reshape(cvImage.shape).astype(np.uint8), iter
       
#########################################################
###    Part C: Normalized Cuts Segmentation Functions
######################################################### 

# TODO:PA2 Fill in this function
def getTotalNodeWeights(W):
    """
    Calculate the total weight of all edges leaving each 
    node.
    
    Parameters:
        W - the m*n x m*n weighted adjecency matrix of a graph
    
    Return:
        d - a vector whose ith component is the total weight
            leaving node i in W's graph.
    """
    d = np.sum(W,axis = 1)
    return d
    #aise NotImplementedError
    
# TODO:PA2 Fill in this function
def approxNormalizedBisect(W, d):
    """
    Find the eigenvector approximation to the normalized cut
    segmentation problem with weight matrix W and diagonal d.
    
    Parameters:
        W - a (n*m x n*m) array of weights (floats)
        d - a n*m vector

    Return:
        y_1 - the second smallest eigenvector of D-W
    """
    D = np.diag(d)
    n,n = W.shape
    I = np.identity(n)

    D = fractional_matrix_power(D,-0.5)
    w,v = eigh(I - D.dot(W).dot(D))

    return np.sum(D*v[:,1],axis = 1)
    #raise NotImplementedError

# TODO:PA2 Fill in this function
def getColorWeights(cvImage, r, sigmaF=5, sigmaX=6):
    """
    Construct the matrix of the graph of the input image,
    where weights between pixels i, and j are defined
    by the exponential feature and distance terms.
    
    Parameters:
        cvImage - the m x n uint8 input image
        r - the maximum distance below which pixels are 
            considered to be connected
        sigmaF - the standard deviation of the feature term
        sigmaX - the standard deviation of the distance term
    
    Return:
        W - the m*n x m*n matrix of weights representing how 
            closely each pair of pixels is connected
    
    """
    #cvImage = np.transpose(cvImage,(1,0,2))
    fig_size = cvImage.shape
    if len(fig_size) == 3:
        [m,n,color] = fig_size
    else:
        [m,n] = fig_size
    
    ind_matrix = np.array([[[np.float(i),np.float(j)] for j in range(n)] for i in range(m)])
    
    if len(fig_size) == 3:
        scipy_color = cvImage.reshape(m*n,color)
    else:
        scipy_color = cvImage.reshape(m*n)
    scipy_ind = ind_matrix.reshape(m*n,2)

    color_dist = np.exp(-distance.pdist(scipy_color,'euclidean')/(sigmaF^2))
    
    dist_temp = distance.pdist(scipy_ind,'euclidean')
    dist = np.exp(-distance.pdist(scipy_ind,'euclidean')/(sigmaX^2))
    dist[dist_temp > r] = 0

    #W = distance.squareform(color_dist)*distance.squareform(dist)
    #np.fill_diagonal(W, 1.0)

    W = np.ones((m*n,m*n))

    ind = 0
    for i in range(0,m*n-1):
        W[i][i+1:] = color_dist[ind:ind+(m*n-1)-i]*dist[ind:ind+(m*n-1)-i]
        W[i+1:,i] = W[i][i+1:]
        #print len(W[i][i+1:]),len(W[i+1:][i]),len(dist[ind:ind+(m*n-1)-i]),i
        ind = ind + (m*n-1)-i

    #raise NotImplementedError
    return W

# TODO:PA2 Fill in this function
def reconstructNCutSegments(cvImage, y, threshold=0):
    """
    Create an output image that is yellow wherever y > threshold
    and blue wherever y < threshold
    
    Parameters:
        cvImage - an (m x n x 3) BGR image
        y - the (m x n)-dimensional eigenvector of the normalized 
            cut approximation algorithm,
        threshold - the cutoff between pixels in the two segments.
        
    Return:
        segmentedImage - an (n x m x 3) image that is yellows
                         for pixels with values above the threshold
                         and blue otherwise.
    """
    m,n,color = cvImage.shape
    y = y.reshape(m,n)
    segmentedImage = np.zeros_like(cvImage)

    #segmentedImage = np.zeros((m*n,3))
    #segmentedImage[y<=threshold] = [255,0,0]
    #segmentedImage[y>threshold] = [0,255,255]
    #segmentedImage = segmentedImage.reshape(m,n,3)
    
    for i in range(m):
        for j in range(n):
            if y[i,j] <= threshold:
                segmentedImage[i,j] = [255,0,0]
            else:
                segmentedImage[i,j] = [0,255,255]
    #raise NotImplementedError
    
    return segmentedImage

    
def nCutSegmentation(cvImage, sigmaF=5, sigmaX=6):
    print("Getting Weight Matrix")
    W = getColorWeights(cvImage, 7)
    print(str(W.shape[0]) + "x" + str(W.shape[1]) + " Weight matrix generated")
    d = getTotalNodeWeights(W)
    print("Calculated weight totals")
    y = approxNormalizedBisect(W, d)
    print("Reconstructing segments")
    segments = reconstructNCutSegments(cvImage, y, 0)
    return segments
    
if __name__ == "__main__":
    pass
    # You can test your code here.