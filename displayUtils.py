import matplotlib.pyplot as plt
import random



def displayImageGrid(images: list, H: int, W: int=0, shuffle=False, figsize=None):
    """
    Display list of images in a grid (H, W) without boundaries. The images MUST be the same size or this will probably look weird.

    Parameters:
    images: List of numpy arrays representing the images. The images should be the same size
    H: Number of rows.
    W: Number of columns.
    """
    
    numImages = len(images)
    
    # Shuffle images before so we can get a good sampling
    if shuffle:
        random.shuffle(images)
    
    # If no width is defined, we assume a single row of images
    if W == 0:
        W = numImages
    
    if numImages < H * W:
        raise ValueError(f"Number of images ({len(images)}) is smaller than given grid size!")
    
    # Shrink figure size if plotting lots of images
    if figsize is None:
        fig = plt.figure(figsize=(W/5, H/5))
    else:
        fig = plt.figure(figsize=figsize)

    for i in range(H * W):
        img = images[i]
        ax = fig.add_subplot(H, W, i+1)
        ax.imshow(img)

        # Remove axis details
        ax.axis('off')
        
        # Adjust the position of the axis for each image
        ax.set_position([i%W/W, 1-(i//W+1)/H, 1/W, 1/H])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

