==============================
pjinoise Requirements Document
==============================

The purpose of this document is to detail the requirements for pjinoise,
a Python image and animation generator. This is an initial take for the 
purposes of planning. There may be additional requirements or non-
required features added in the future.


Purpose
-------
The purpose of pjinoise is to generate images and animations that show 
the complexity of simple patterns interacting.


Functional Requirements
-----------------------
The following are the functional requirements for pjinoise:

*   Generate raster images of arbitrary size.
*   Generate animations of arbitrary size.
*   Have multiple types of image content generators, including an 
    octave Perlin noise generator.
*   Be able to apply filters to the image content that modify that 
    content.
*   Be able to generate the image content in layers that interact 
    with each other through operations such as difference and 
    multiply.
*   Store the configuration of the image, so that the image can be 
    recreated at different scales.


Technical Requirements
----------------------
The following are the technical requirements for pjinoise:

*   pjinoise must contain its own implementation of the Perlin 
    noise algorithm.
*   pjinoise should use numpy and favor vectorized processing 
    where possible.


Security Requirements
---------------------
The following are the security requirements for pjinoise:

*   No objects will be serialized to disk in an executable form, 
    instead they shall be stored as data that can be validated 
    before reconstructing the original object.


Design Discussion
-----------------
The following is a deeper discussion of certain aspects of the 
pjinoise design. This primarily exists as a place to talk through 
design challenges in order do find solutions. It is not intended 
to be comprehensive nor even completely accurate to the final 
design.


Core Image Creation Steps
~~~~~~~~~~~~~~~~~~~~~~~~~
The following steps are used to generate the final image or 
animation:

1.  Initial setup, including building the following:
    a.  Layer generators
    b.  Layer filters
    c.  Image filters
2.  Review the generators and filters to determine if the the 
    size of the image or other configuration needs to be changed.
3.  Update the configuration with any changes.
4.  Generate a layer of noise.
5.  Apply any layer filters to the layer contents.
6.  Combine the layers based on the operators, such as diference 
    or multiply, assigned to those layers.
7.  Apply any image filters to the image.
8.  
