#########################################################################
#                                                                       #
#    Author Yannick Paul Klose                                          #
#    Year   2019                                                        #
#                                                                       #
#########################################################################

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import random

def image_generator(background_path):
    
    # does image augmentation on a given image (as datapath)
    
    background = Image.open(background_path).convert("RGBA")    
    x, y = background.size
    
    # random mirrowing
    random_var = random.randint(0,1)
    if random_var == 1:
        background = background.transpose(Image.FLIP_LEFT_RIGHT)
        
    # slight rotation
    rotate_angle = random.randint(-10, 10)
    background = background.rotate(rotate_angle, expand = False)

    
    # set random blur
    minimum_blur = -1.0
    maximum_blur = 1.5
    random_blur = random.randint(minimum_blur*100, maximum_blur*100)/100
    background = background.filter(ImageFilter.GaussianBlur(random_blur))

    
    # set random brightness
    minimum_brightness = 0.5
    maximum_brightness = 1.5
    random_brightness = random.randint(minimum_brightness*100, maximum_brightness*100)/100
    background = ImageEnhance.Brightness(background).enhance(random_brightness)
    

    return background


