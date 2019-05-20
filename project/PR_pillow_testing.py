# pillow testing


from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import random


background_path = "test_images/background.jpg"
object_path = "test_images/model.png"

def overlay(background_path, object_path):
    
    background = Image.open(background_path).convert("RGBA")
    foreground = Image.open(object_path).convert("RGBA")
    x1, y1 = foreground.size
    x2, y2 = background.size

    # random scaling
    minimum_scaling = 0.6
    maximum_scaling = 0.9
    resize_factor = random.randint(minimum_scaling*10, maximum_scaling*10)/10
    foreground.thumbnail((x1*resize_factor, y1*resize_factor))

    # random rotate factor
    rotate_angle = random.randint(-45, 45)
    foreground = foreground.rotate(rotate_angle, expand = True)
    x1, y1 = foreground.size

    # set random position
    rand_x = random.randint(0, x2-x1)
    rand_y = random.randint(0, y2-y1)
    
    # set random blur
    minimum_blur = 0.0
    maximum_blur = 1.2
    random_blur = random.randint(minimum_blur*100, maximum_blur*100)/100
    foreground = foreground.filter(ImageFilter.GaussianBlur(random_blur))

    # set random brightness
    minimum_brightness = 0.6
    maximum_brightness = 1.3
    random_brightness = random.randint(minimum_brightness*100, maximum_brightness*100)/100
    foreground = ImageEnhance.Brightness(foreground).enhance(random_brightness)

    # pase foreground image on background image
    background.paste(foreground, (rand_x, rand_y), foreground)
    center_x = int(rand_x+x1/2)
    center_y = int(rand_y+y1/2)
    
    # save the image (for testing purpose)
    #background.save("test_image.png", format="png")

    return background, center_x, center_y, rand_x, rand_y, x1, y1

#image, center_x, center_y, rand_x, rand_y, x1, y1 = overlay(background_path, object_path)
#print(image)

