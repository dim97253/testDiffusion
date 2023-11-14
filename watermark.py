from PIL import Image

def add_watermark(original_image):
    watermark_image = Image.open('logo.png')
    watermark_position = (original_image.width - watermark_image.width, 0)
    new_size = tuple(int(dim * 0.7) for dim in watermark.size)
    watermark = watermark.resize(new_size, Image.ANTIALIAS)
    # Paste watermark image on original image at the calculated position
    original_image.paste(watermark_image, watermark_position, watermark_image)

    return original_image
