from PIL import Image


image = Image.open('watermarks/1 (2).png')
alpha = image.split()[-1]
width, height = image.size
print(width, height)
print(alpha)
image.putalpha(alpha)
alpha.save('test.png')
image.save('aaddingalofa.png')