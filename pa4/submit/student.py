# Please place imports here.
# BEGIN IMPORTS
import numpy as np
from numpy.linalg import inv
import scipy
from scipy.ndimage.filters import correlate
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- 3 x N array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    new_images = np.array(images)
    size = new_images.shape
    channel = size[3]
    N = size[0]
    height = size[1]
    width = size[2]
    if channel == 1: # grey scale
        I = new_images.reshape(N,height*width)
        G = inv(lights.dot(lights.transpose())).dot(lights).dot(I)
        albedo = np.sqrt(np.sum(G**2,axis=0))
        temp = albedo.copy()
        temp[albedo==0] = 1.0
        normals = G/temp
        normals[0,albedo < 1e-7] = 0
        normals[1,albedo < 1e-7] = 0
        normals[2,albedo < 1e-7] = 0
        albedo[albedo <1e-7] = 0
        normals = normals.transpose().reshape(height,width,3)
        albedo = albedo.reshape(height,width,1)
    elif channel == 3:
        I = new_images.reshape(N,height*width,3)
        albedo = np.zeros((3,height*width))
        for i in range(2,-1,-1):
            G = inv(lights.dot(lights.transpose())).dot(lights).dot(I[:,:,i])
            albedo[i,:] =  np.sqrt(np.sum(G**2,axis=0))
        albedo_l2 = np.sqrt(np.sum(albedo**2,axis=0))
        albedo[0,albedo_l2 <1e-7] = 0
        albedo[1,albedo_l2 <1e-7] = 0    
        albedo[2,albedo_l2 <1e-7] = 0
        temp = albedo[0,:]
        temp[albedo[0,:]==0] = 1.0
        normals = G/temp
        normals[0,albedo_l2 < 1e-7] = 0
        normals[1,albedo_l2 < 1e-7] = 0
        normals[2,albedo_l2 < 1e-7] = 0        
        normals = normals.transpose().reshape(height,width,3)
        albedo = albedo.transpose().reshape(height,width,3)     
           
    return albedo,normals

    #aise NotImplementedError()


def pyrdown_impl(image):
    """
    Prefilters an image with a gaussian kernel and then downsamples the result
    by a factor of 2.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/16 [ 1 4 6 4 1 ]

    Functions such as cv2.GaussianBlur and
    scipy.ndimage.filters.gaussian_filter are prohibited.  You must implement
    the separable kernel.  However, you may use functions such as cv2.filter2D
    or scipy.ndimage.filters.correlate to do the actual
    correlation / convolution.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Downsampling should take the even-numbered coordinates with coordinates
    starting at 0.

    Input:
        image -- height x width [x channels] image of type float32.
    Output:
        down -- ceil(height/2) x ceil(width/2) [x channels] image of type
                float32.
    """
    K = 1.0/16.0*np.array([[1,4,6,4,1]])
    size = image.shape
    width = size[1]
    height = size[0]
    down = np.zeros_like(image)
    if len(size) == 3: # more than 1 channel
        channel = size[2]
        for i in range(channel):
            down[:,:,i] = correlate(image[:,:,i],K,mode = 'mirror')
        K = K.transpose()
        for i in range(channel):
            down[:,:,i] = correlate(down[:,:,i],K,mode = 'mirror')

        w = [i for i in range(0,width,2)]
        h = [i for i in range(0,height,2)]
        down = down[h,:,:]
        down = down[:,w,:]
    else:
        down = correlate(image,K,mode = 'mirror')
        K = K.transpose()
        down = correlate(down,K,mode = 'mirror')

        w = [i for i in range(0,width,2)]
        h = [i for i in range(0,height,2)]
        down = down[h,:,]
        down = down[:,w,]       
    
    return down

    #raise NotImplementedError()


def pyrup_impl(image):
    """
    Upsamples an image by a factor of 2 and then uses a gaussian kernel as a
    reconstruction filter.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/8 [ 1 4 6 4 1 ]
    Note: 1/8 is not a mistake.  The additional factor of 4 (applying this 1D
    kernel twice) scales the solution according to the 2x2 upsampling factor.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Upsampling should produce samples at even-numbered coordinates with
    coordinates starting at 0.

    Input:
        image -- height x width [x channels] image of type float32.
    Output:
        up -- 2 height x 2 width [x channels] image of type float32.
    """
    #raise NotImplementedError()
    K = 1.0/8.0*np.array([[1,4,6,4,1]])
    size = image.shape
    width = size[1]
    height = size[0]
    if len(size) == 3:
        channel = size[2]
        up = np.zeros((height*2,width*2,channel))
        up[1::2,1::2,:] = image 
        for i in range(channel):
            up[:,:,i] = correlate(up[:,:,i],K,mode = 'mirror')
        K = K.transpose()
        for i in range(channel):
            up[:,:,i] = correlate(up[:,:,i],K,mode = 'mirror')
    else:
        up = np.zeros((height*2,width*2))
        up[1::2,1::2] = image 
        up = correlate(up,K,mode = 'mirror')
        K = K.transpose()
        up = correlate(up,K,mode = 'mirror')      
    return up          



def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    
    height,width,coor = points.shape
    new_points = points.flatten().reshape(height*width,coor).transpose()
    new_points = np.insert(new_points,coor,1,axis = 0)

    projections = K.dot(Rt).dot(new_points)
    projections = (projections[0:coor-1,:]/projections[coor-1,:]).transpose().reshape(height,width,coor-1)
    return projections
    #raise NotImplementedError()


def unproject_corners_impl(K, width, height, depth, Rt):
    """
    Undo camera projection given a calibrated camera and the depth for each
    corner of an image.

    The output points array is a 2x2x3 array arranged for these image
    coordinates in this order:

     (0, 0)      |  (width, 0)
    -------------+------------------
     (0, height) |  (width, height)

    Each of these contains the 3 vector for the corner's corresponding
    point in 3D.

    Tutorial:
      Say you would like to unproject the pixel at coordinate (x, y)
      onto a plane at depth z with camera intrinsics K and camera
      extrinsics Rt.

      (1) Convert the coordinates from homogeneous image space pixel
          coordinates (2D) to a local camera direction (3D):
          (x', y', 1) = K^-1 * (x, y, 1)
      (2) This vector can also be interpreted as a point with depth 1 from
          the camera center.  Multiply it by z to get the point at depth z
          from the camera center.
          (z * x', z * y', z) = z * (x', y', 1)
      (3) Use the inverse of the extrinsics matrix, Rt, to move this point
          from the local camera coordinate system to a world space
          coordinate.
          Note:
            | R t |^-1 = | R' -R't |
            | 0 1 |      | 0   1   |

          p = R' * (z * x', z * y', z, 1)' - R't

    Input:
        K -- camera intrinsics calibration matrix
        width -- camera width
        height -- camera height
        depth -- depth of plane with respect to camera
        Rt -- 3 x 4 camera extrinsics calibration matrix
    Output:
        points -- 2 x 2 x 3 array of 3D points
    """

    R = Rt[0:3,0:3]
    t = Rt[0:3,3]
    points = np.array([[0,0,1],[width,0,1],[0,height,1],[width,height,1]]).transpose()
    points = inv(K).dot(points)*depth
    points = np.transpose(R.transpose().dot(points[0:3,:])) - np.sum(R.transpose()*t,axis = 1)
    points = points.reshape(2,2,3)

    return points
    #raise NotImplementedError()


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the channel-interleaved, column
    major order (more simply, flatten on the transposed patch).
    For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x112, x211, x212, x121, x122, x221, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- height x width x (channels * ncc_size**2) array
    """
    height,width,channel = image.shape
    half_ncc = (ncc_size)/2
    normalized = np.zeros((height,width,channel*(ncc_size**2)))
    for i in range(0,height):
        for j in range(0,width):
            if i+half_ncc < height and i-half_ncc >= 0 and j+half_ncc < width and j-half_ncc >= 0:
                for k in range(0,channel):
                    patch = image[(i-half_ncc):(i+half_ncc+1),(j-half_ncc):(j+half_ncc+1),k].flatten()
                    patch = patch - np.mean(patch)
                    normalized[i,j,k*len(patch):(k+1)*len(patch)] = patch
                norm = np.sqrt(np.sum(normalized[i,j,:]**2))
                if norm < 1e-6:
                    normalized[i,j,:] = 0
                else:
                    normalized[i,j,:] = normalized[i,j,:]/norm

    return normalized

    #raise NotImplementedError()


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    ncc = np.sum(image1*image2,axis = 2)
    return ncc
    #raise NotImplementedError()
