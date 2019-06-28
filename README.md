# Imaging-System-3D-Noise-
This is a python implementation of the 3D noise model originally used by Center for Night Vision and Electro-Optics to analyze spatio-temporal noise components in imaging systems.

Reference: J. D' Agostino and C. Webb, "Three-dimensional analysis framework and measurement methodology for imaging system noise," *Proceedings of SPIE* vol. 1488.

These methods not only provide a simple means of analyzing noise in an imaging system in the most general possible way but also of adding noise of any desired spatio-temporal distribution to a set of real or synthetic images.

The module includes three straightforward methods:

## make_tiff_data_cube:
Create a cube of data from a directory of .tif files. This data should comprise a stack of image data taken under constant illumination.

## get_3dnoise:
Calculate all spatio-temporal noise components from the image data cube to obtain complete imaging system noise characteristics.

## set_3dnoise:
Add synthetic noise of any spatio-temporal distribution to pre-existing synthetic or real images. This might be useful, for example, for creating or augmenting image data for use in training a computer vision AI model
