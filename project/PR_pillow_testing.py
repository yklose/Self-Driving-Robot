# pillow testing


from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import random



def overlay_output(background_path, paste):
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
    n_of_patterns = 1
    scale = 0.3
    imgWidth ,imgHeight = img.size
    fitWidth = imgWidth * scale # fitWidth could be a float!
    #Axis = []
    for n in range(n_of_patterns):
        random_fit_width = int(fitWidth * random.uniform(0.2,1.0))
        pattern = generate_pattern(random_fit_width)# generate_pattern() # pattern is also a PIL.image file
        patternWidth, patternHeight = pattern.size
        pattern.save(os.path.join('transformed_pattern.png'))
        position_x = random.randint(0,imgWidth - patternWidth)
        position_y = random.randint(0,imgHeight - patternHeight)

        if paste:
            img.paste(pattern,box =(position_x,position_y),mask = pattern) 
        # the third argument serves as a mask, making white = 0 transparent
        # Axis.append ([position_x,position_y,position_x + patternWidth,position_y + patternHeight])
        center_x = int(position_x + 0.5*patternWidth,position_y+0.5*patternHeight)

    return img , center_x, center_y, position_x, position_y, patternWidth, patternHeight

def generate_pattern(random_fit_width):
    """The function takes random_fit_Width, which could be a float!
    The function dose the following:
        1. Read the pattern file (must be PNG to have transparency)
        2. Do random transformation on it.
        3. Return a transformed PIL.image file
    """
    pattern = Image.open('pattern.png').convert("RGBA")
    patternWidth, patternHeight = pattern.size
    patternHeight = int(random_fit_width / patternWidth * patternHeight)
    patternWidth = random_fit_width
    #pattern = pattern.resize((patternWidth, patternHeight))
    #pattern.ToTensor()
    pattern = functional.resize(pattern,(patternHeight,patternWidth))
    transform = Compose([
        ColorJitter(brightness= (0.5,1.1), contrast = (0.5,1),
         saturation= (0.75,1), hue= 0.05),
        #RandomApply((RandomResizedCrop(size = (patternWidth, patternHeight),scale = (0.75,1.0)),),p = 0.25),
        RandomAffine(degrees=45),
        #RandomApply((RandomRotation(degrees = 30), ), p = 0.75),
        #RandomPerspective(distortion_scale=0.5, p=0.5,)
    ])    

    return transform(pattern)

'''
def overlay(background_path, paste):
    
    object_path = "test_images/model.png"
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
    rand_x = random.randint(0, abs(x2-x1))
    rand_y = random.randint(0, abs(y2-y1))
    
    # set random blur
    minimum_blur = 0.0
    maximum_blur = 1.2
    random_blur = random.randint(minimum_blur*100, maximum_blur*100)/100
    foreground = foreground.filter(ImageFilter.GaussianBlur(random_blur))

    # affine transform!
    
    # set random brightness
    minimum_brightness = 0.6
    maximum_brightness = 1.3
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

    
    # save the image (for testing purpose)
    #background.save("test_image.png", format="png")

    return background, center_x, center_y, rand_x, rand_y, x1, y1

#image, center_x, center_y, rand_x, rand_y, x1, y1 = overlay(background_path, object_path)
#print(image)
'''
