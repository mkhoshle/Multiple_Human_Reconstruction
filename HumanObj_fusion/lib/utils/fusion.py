import torch
import numpy as np
import cv2

def Laplacian_pyramid(image_list):
    # Generate Laplacian pyramid for each image
    laplacian_pyramids = []
    for image in image_list:
        image = image.detach().cpu().numpy()
        pyramid = [image]
        temp = image.copy()
        for _ in range(5):  # Adjust the number of pyramid levels as per your requirement
            temp = cv2.pyrDown(temp)
            pyramid.append(temp)
        laplacian_pyramids.append(pyramid)

    # Blend the Laplacian pyramid levels
    blended_pyramid = []
    for level in range(5):  # Same number of pyramid levels for all images
        blended_level = np.zeros_like(laplacian_pyramids[0][level])
        for pyramid in laplacian_pyramids:
            blended_level += pyramid[level]
        blended_pyramid.append(blended_level / len(image_list))

    # Reconstruct the blended image from the pyramid
    blended_image = blended_pyramid[4]
    for i in range(3, -1, -1):
        blended_image = cv2.pyrUp(blended_image)
        blended_image = cv2.add(blended_image, blended_pyramid[i])

    return torch.from_numpy(blended_image).to('cuda:0')
