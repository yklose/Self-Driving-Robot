# tester

from PIL import Image, ImageDraw

def allocate_center(xpoint, ypoint):
    image_dir = "test.png"
    image = Image.open(image_dir).convert("RGBA")
    
    draw = ImageDraw.Draw(image)
    draw.point((xpoint, ypoint), 'red')
    
    image.save("center_e2.png", format="png")
    
    
allocate_center(490, 248)