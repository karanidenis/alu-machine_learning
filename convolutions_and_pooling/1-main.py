import numpy as np
import matplotlib.pyplot as plt
# Define the path to the MNIST dataset file
train_images_path = '/content/drive/MyDrive/Colab-Notebooks/Images/train-images-idx3-ubyte'

# Load training images
with open(train_images_path, 'rb') as f:
    # Skip the header (first 16 bytes) and reshape the data
    train_images = np.fromfile(
        f, dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    # Select the first 10 images from the original dataset for testing
    small_dataset = train_images[:500]

convolve_grayscale_valid = __import__(
    '0-convolve_grayscale_valid').convolve_grayscale_valid


if __name__ == '__main__':
    images = small_dataset
    print(images.shape)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_valid(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
