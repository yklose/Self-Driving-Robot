#########################################################################
#                                                                       #
#    Author Yannick Paul Klose                                          #
#    Year   2019                                                        #
#                                                                       #
#########################################################################

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import random
import PR_image_generator

def overlay(background_path, object_path, paste, after_training):
    
    # this functions paste a given object on a background 
    # random scaling, rotation, position, noise and brightness
    
    if after_training:
        rand_select = random.randint(0, 3)
        # select background image from Coco in 33 percent of the time!
        if rand_select == 1:                
            background = Image.open(background_path).convert("RGBA") 
        else:
            rand_image = random.randint(1, 54)
            datapath_buffer = "extended_data/IMG_" + str(rand_image) + ".jpeg"
            background = PR_image_generator.image_generator(datapath_buffer)
    else:    
        background = Image.open(background_path).convert("RGBA")        
    foreground = Image.open(object_path).convert("RGBA")
    x1, y1 = foreground.size
    x2, y2 = background.size


    # random scaling
    minimum_scaling = 0.08
    maximum_scaling = 0.18
    resize_factor = random.randint(minimum_scaling*100, maximum_scaling*100)/100
    foreground.thumbnail((x1*resize_factor, y1*resize_factor))

    # random rotate factor
    rotate_angle = random.randint(-45, 45)
    foreground = foreground.rotate(rotate_angle, expand = True)
    x1, y1 = foreground.size

    # set random position
    rand_x = random.randint(0, abs(x2-x1))
    rand_y = random.randint(0, abs(y2-y1))
    
    # set random blur
    minimum_blur = 0.0
    maximum_blur = 1.5
    random_blur = random.randint(minimum_blur*100, maximum_blur*100)/100
    foreground = foreground.filter(ImageFilter.GaussianBlur(random_blur))
    
    # set random brightness
    minimum_brightness = 0.5
    maximum_brightness = 1.0
    random_brightness = random.randint(minimum_brightness*100, maximum_brightness*100)/100
    foreground = ImageEnhance.Brightness(foreground).enhance(random_brightness)

    if (paste):
        # pase foreground image on background image
        background.paste(foreground, (rand_x, rand_y), foreground)
        center_x = int(rand_x+x1/2)
        center_y = int(rand_y+y1/2)
    else:
        center_x = 0
        center_y = 0


    return background, center_x, center_y, rand_x, rand_y, x1, y1


