"""
template
~~~~~~~

Working file for creating images with the pjinoise module.
"""
from pjinoise import filters as f
from pjinoise import model as m
from pjinoise import operations as op
from pjinoise import pjinoise as pn
from pjinoise import sources as s


# pjinoise version check.
assert pn.__version__ == '0.3.1'


# Basic image file configuration and other commonly used values.
filename = 'work.jpg'
filetype = 'JPEG'
colorspace = 'RGB'
size = (1, 720, 1280)


# Layers.
layer = m.Layer(**{
    'source': None,
    'blend': None,
    'filters': [],
    'mask': None,
    'mask_filters': [],
})


# Image.
conf = m.Image(**{
    'source': layer,
    'size': size,
    'filename': filename,
    'format': filetype,
    'mode': colorspace,
})

# Create image.
pn.main(False, conf)
