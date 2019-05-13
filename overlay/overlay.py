import os
from PIL import Image
from torchvision.transforms import *
import random

def overlay_output(img, n_of_patterns = 5, scale = 0.3):
    """The function dose the following:
        1. Rescale the generated pattern and make it suit he img.
        2. Find a random axis to place the pattern.
        3. Paste the pattern to the selected place.
    Args:
        img: an image module in PIL img = PIL.Image.open()
        n_of_patterns: an int.
        scale: a float, indicating possible scaling range of the pattern.
    Return:
        Whatever suit the training input function.
        And a list indicating x, y, Width, Height
    """
    imgWidth ,imgHeight = img.size
    fitWidth = imgWidth * scale # fitWidth could be a float!
    Axis = []
    for n in range(n_of_patterns):
        pattern = Image.open('pattern.png').convert("RGBA")# generate_pattern() # pattern is also a PIL.image file
        patternWidth, patternHeight = pattern.size
        random_fit_width = int(fitWidth * random.uniform(0.2,1.0))
        patternHeight = int(random_fit_width / patternWidth * patternHeight)
        patternWidth = random_fit_width
        pattern = pattern.resize((patternWidth, patternHeight))

        position_x = random.randint(0,imgWidth - patternWidth)
        position_y = random.randint(0,imgHeight - patternHeight)

        img.paste(pattern,box =(position_x,position_y),mask = pattern) 
        # the third argument serves as a mask, making white = 0 transparent
        Axis.append ([position_x,position_y,position_x + patternWidth,position_y + patternHeight])

    return img , Axis

def generate_pattern():
    """The function dose the following:
        1. Read the pattern file (must be PNG to have transparency)
        2. Do random transformation on it.
        3. Return a transformed PIL.image file
    """
    pattern = Image.open('pattern.png')
    patternWidth, patternHeight = pattern.size
    pattern.ToTensor()
    transform = Compose([
        ColorJitter(brightness= (0.5,1.1), contrast = (0.5,1),
         saturation= (0.75,1), hue= 0.05),
        RandomApply((RandomResizedCrop(size = (patternWidth, patternHeight),scale = (0.75,1.0)),),
         p = 0.25),
        # RandomApply((RandomAffine(), ),
        # p = 0.5),
        RandomApply((RandomRotation(degrees = 15), ),
        p = 0.75),
        # RandomPerspective(distortion_scale=0.5, p=0.5,)
    ])

    pattern.transform()
    pattern.ToPILImage()

    return pattern
    
