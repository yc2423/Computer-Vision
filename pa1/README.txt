We pass all test cases.


- Setting for low pass image:
image: dog
sigma: 6.0
kernel size: 5

- Setting for high pass image:
image: cat
sigma: 5.0
kernel size: 6

In order to make the program run faster, we choose small kernel sizes.
We also use the setting described above to create the hybrid image, and the mix-in ratio is 0.6.
In order to make the hybrid image looks brighter, we change the following two lines:

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio 

to
    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 5 * mixin_ratio