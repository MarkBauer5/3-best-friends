import matplotlib.pyplot as plt
import random



def displayImageGrid(images: list, H: int, W: int=0, labels: list=None, figsize=None, title=''):
    """
    Display list of images in a grid (H, W) without boundaries. The images MUST be the same size or this will probably look weird.

    Arguments:
        images: List of numpy arrays representing the images. The images should be the same size
        H: Number of rows.
        W: Number of columns.
        figsize: The figure size of the output plot
        title: A title for the plot
    """
    
    numImages = len(images)
    
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
        label = labels[i] if labels is not None else ''
            
        ax = fig.add_subplot(H, W, i+1)
        ax.imshow(img)

        # Remove axis details
        ax.axis('off')
        
        # Adjust the position of the axis for each image slightly to make room for the label
        # ax.set_position([i%W/W, 1-(i//W+1)/H + 0.2, 1/W, 1/H - 0.05])
        
        # Adjust the position of the axis for each image to make more room for the label
        ax.set_position([i%W/W, 1-(i//W+1)/H + 0.05, 1/W, 1/H - 2])
        
        # Add label below the image
        ax.set_title(label, fontsize=10, pad=6, y=-0.25) # pad adds space between the image and the label
        
        # Add label below the image
        # ax.set_title(label, fontsize=10, y=-0.2)
        

    plt.suptitle(title)
    plt.subplots_adjust(wspace=0, hspace=0, top=0.95)
    plt.show()

