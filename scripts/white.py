from PIL import Image
import numpy as np

# Create a white image (400x400 pixels)
white_img = Image.new('RGB', (400, 400), color = (255, 255, 255))

# Save the image
white_img.save('white.jpg')

print("white.jpg has been created successfully!")