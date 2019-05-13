from overlay import *
from PIL import Image, ImageDraw
import numpy as np

def main():
    img = Image.open('000000000049.jpg')

    overlaid, Axis = overlay_output(img)
    print (Axis)
    draw = ImageDraw.Draw(overlaid)
    pattern_number = len(Axis)
    for i in range(pattern_number):
        
        draw.rectangle(Axis[i],width=1)

    overlaid.save(os.path.join('overlaied.jpg'))

if __name__ == "__main__":
    main()