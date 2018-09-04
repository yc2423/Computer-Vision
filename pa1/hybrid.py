import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    fig_size = img.shape
    #img = img.astype(float)
    #kernel = kernel.astype(float)
    height = fig_size[0]
    width = fig_size[1]
    new_img = np.zeros_like(img)
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
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    m,n = kernel.shape
    kernel = kernel.flatten()[::-1]
    kernel = kernel.reshape(m,n)
    new_img = cross_correlation_2d(img,kernel)
    return new_img

    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    kernel = np.zeros((width,height))
    #sigma = float(sigma)
    for i in range(0,width/2+1):
        for j in range(0,height/2+1):
            x = float(i - width/2)
            y = float(j - height/2)
            kernel[i][j] = 1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))
            kernel[i,height-1-j] = kernel[i,j]
        kernel[width-1-i,:] = kernel[i,:]
    kernel = kernel/np.sum(kernel,axis=None)
    return kernel
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma,size,size)
    new_img = convolve_2d(img,kernel)
    return new_img
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    low_fig = low_pass(img,sigma,size)
    new_img = img - low_fig
    return new_img
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 5 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


