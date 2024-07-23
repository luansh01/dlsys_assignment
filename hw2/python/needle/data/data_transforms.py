import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            img = np.flip(img,axis=1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        img_shape = img.shape
        padded_img = np.pad(img, pad_width = ((self.padding, self.padding), (self.padding, self.padding),(0,0) ), mode='constant',constant_values=0)
        # Perform cropping
        start_width = self.padding  + shift_x
        end_width = img_shape[0]+self.padding + shift_x 
        start_height = self.padding  + shift_y
        end_height = img_shape[1]+self.padding + shift_y 
        padded_img = padded_img[start_width:end_width,start_height:end_height]
        return padded_img
        ### END YOUR SOLUTION
