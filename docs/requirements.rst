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


Linking Layers to Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
During step 6 of the core image creation, pjinoise needs to know:

*   The layers to combine.
*   The operations to use for the combination.

The layers are determined by the order they appear in the iterable 
they are stored in. The operation can also be done that way, but it 
it's a little awkward to store them in separate iterables. So, the 
iterable that contains the layers should be structured as:

    ```
    [
        [operation, layer],
        [operation, layer],
        ...
    ]
    ```

Then when iterating through the layers to perform the combinations, 
the first combination should be a replace operation on a zeros array.


Core, CLI, and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The original implementation of pjinoise assumed that the layers of an 
image all used the same type of generator object. This simplified the 
CLI interface because it allowed the arguments to be applied across 
the generators, rather than having to worry about how to configure 
individual generators.

The refactoring being done in pjinoise.core changes that. Now, I want 
to assume that the layers will be made by different generators using 
different blend modes. This seems to complicate the CLI because now 
I need to worry about how a user would input the configuration for 
each of the generators separately. While that's doable through 
CLI arguments, it's messy enough that I think the main way to 
interact with pjinoise is going to need to move to input files. 

However, I don't want to completely foreclose on the possibility of a 
useful CLI. So, what would a CLI for core look like?

Setting general configuration like the size of the output image and 
the filename I want to save to doesn't change:

    ```
    pjinoise.py -s 1280 720 50 -l 0 0 50 -o spam.mp4
    ```

The append action in argparse can be used to allow multiple generators 
to be configured from the command line. The arguments for the generator 
would be passed as a colon delimited list with the type of generator:

    ```
    pjinoise.py -s 1280 720 50 -l 0 0 50 -n LineNoise_255:0:h:128:
    ioq_replace -n LineNoise_255:0:v:128:iq_difference -o spam.mp4
    ```

The difficulty is how to handle the filters for the layer. Since I'm 
defining them with the generator, I don't need to colon to determine 
which layers the filter applies to any more. Bang probably isn't doing 
anything, so that can be a delimiter character within the filter 
definition. So, maybe something like:

    ```
    pjinoise.py -s 1280 720 50 -l 0 0 50 -n LineNoise_255:0:h:128:
    ioq_replace -n LineNoise_255:0:v:128:iq_skew:.1!rotate90:r_
    difference -o spam.mp4
    ```

There the underscore is delimiting the filter section from the 
generator argument and blend mode argument. It's a bear to type, but 
I think it works well enough as a CLI solution, with the intention 
that most configuration would be done with configuration files.

There is one last argument to add: location. Since it contains multiple 
elements, it needs two layers of delimiting, too. It's not a part of 
the attributes of the generator, so it doesn't make sense in the 
attributes section. So, I guess it's also underscore delimited, with 
colons doing internal delimiting.

    ```
    pjinoise.py -s 1280 720 50 -l 0 0 50 -n LineNoise_255:0:h:128:
    ioq_0:0:50__replace -n LineNoise_255:0:v:128:iq_0:0:0_skew:.1!
    rotate90:r_difference -o spam.mp4
    ```

In this example empty fields are still marked with underscores. That's 
easier to parse, but harder to type and read. I'll probably start by 
requiring it, but try and figure out a better way eventually.


Core and Color
~~~~~~~~~~~~~~
The original implementation of pjinoise only injects color in as a 
final colorization step. This is a bit awkward because it doesn't 
allow for multicolored images. It would be nice if core would allow 
for multicolor mapping.

The trick is figuring out when to insert that mapping. The layer 
generation is all grayscale. That is to speed up processing since 
processing a three value color is three times more expensive than 
processing a single value. However, future colorization can take 
advantage of this by treating the grayscale layers as color channels. 
allowing for fairly arbitrary colorization ability that is based on 
layer content.

However, that's a rather complicated addition. For now, I'm going 
to focus on reimplementing the colorization ability from the 
original implementation, and I'll come back to adding multicolor 
support.

Given that I'm going with a final colorization, that seems to be 
best placed in the image bake. Color will me part of the image 
configuration.


Core and UI
~~~~~~~~~~~
The original UI for pjinoise was basically just a log of actions. It 
only updated when an action was complete, and it quickly rolled off 
the screen. I'd like the UI for core to avoid those problems.

First, though, what do I want the UI for core to look like? How about 
this:

    ```
    PJINOISE: Pattern and Noise Generation
    ┌       ┐
    │██░░░░░│
    └       ┘
    00:01:23 Generating images...
    ```

That seems doable. I might be able to write that on my own, but I do 
have the interfaces from blackjack and life that I can look at, too. 
It might be best to just go with something ncurses-like. Though, the 
problem with using blessed is that it takes over the full terminal 
and goes away once the program ends. I'd like to keep the display so 
I can see how long the generation took.

That probably means interprocess messaging. Why process and not thread? 
The image generation is CPU bound not IO bound, so concurrency should 
be process-focused not thread-focused. Granted, the UI isn't CPU bound, 
but I really only want to use one type of concurrency, I think. So 
interprocess communication it is.

If I need interprocess messaging, I'll need a message protocol. I can 
keep the code in the functions pretty abstract, and maybe even use a 
decorator rather than inserting the messaging code into the functions. 
However it's done, the UI updater will need to have the ability to 
do these things based on messages:

*   Draw initial state.
*   Update status message.
*   Update progress bar.
*   Terminate.

The messages should be tuples with the structure:

    ```
    (command, args)
    ```

The arguments will vary depending on the command. The breakdown is as 
follows:

------- --- ----------- ------------
CMD     ID  ARGS        DESCRIPTION
------- --- ----------- ------------
INIT    0   None        Start UI.
STATUS  1   str         Update the status msg without updating progress.
PROG    2   str         Update progress and the status message.
END     F   str         Update the status message and terminate.
------- --- ----------- ------------

