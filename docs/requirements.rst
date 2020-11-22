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


MULTI-IMAGE OPERATIONS
~~~~~~~~~~~~~~~~~~~~~~
The original process for image generation described in "Core Image 
Creation Steps" above had a problem. You couldn't limit which lower 
layers were affected by a higher layer. It was all or nothing. This 
was solved by allowing for the creation of multiple images. Now layers 
can be grouped, so you can blend a few layers together before blending 
the result with other layers.

This solved the problems, but highlighted a couple of other problems:

*   You now can't run filters on the final merged image,
*   You can't easily duplicate noise across multiple images.

The second one isn't a new problem. It's been true all along. I'd just 
like to solve it as I solve the first one. So, the requirements are:

1.  pjinoise can run filters on the result of multiple blended images.
2.  pjinoise has noise generators only generate noise once when used 
    in multiple layers.
3.  pjinoise has generators that will duplicate the results of 
    multiple blended layers or images.

OK, so how do I address these?


1: FILTERS ON MULTIPLE BLENDED IMAGES
#####################################
Requirement 1 potentially has two parts:

A.  Running filters after image blending,
B.  Creating subgroups of images for blending.

An example use case:

    Evanesco is creating a final image out of images A, B, and C. 
    They wish to adjust the contrast of the blended image of B and 
    C before that is blended with A. They then want to blur the 
    blended image A, B, and C.

Achieving this will probably need a huge change to the image generation 
process. Today the process for generating the image would be:

1.  Generate A.
2.  Run filters on A.
3.  Generate B.
4.  Run filters on B.
5.  Blend A and B to make AB.
6.  Generate C.
7.  Run filters on C.
8.  Blend AB with C to make ABC.

The new process would need to look like this:

1.  Generate A.
2.  Run A filters on A.
3.  Generate B.
4.  Run B filters on B.
5.  Generate C.
6.  Run C filters on C.
7.  Blend B and C to make BC.
8.  Run BC filters on BC.
9.  Blend A and BC to make ABC.
10. Run ABC filters on ABC.

That means there are a couple of problems to solve:

1.  How do I indicate how to group the images for blending?
2.  How do I indicate the filters that should apply to that group?

I think the following JSON configuration would solve both.

    ```
    {
        "Version": "0.1.1",
        "ImageConfig": {
            "type": "ImageConfig",
            "size": [1, 720, 1280],
            "layers": [
                {
                    "type": "LayerConfig",
                    "generator": "ring",
                    "mode": "difference",
                    "location": [0, 0, 0],
                    "filters": [],
                    "args": [256, 64, "l"]
                },
                {
                    "type": "ImageConfig",
                    "size": [1, 720, 1280],
                    "layers": [
                        {
                            "type": "LayerConfig",
                            "generator": "ring",
                            "mode": "difference",
                            "location": [0, 0, 0],
                            "filters": [],
                            "args": [128, 64, "l"]
                        },
                        {
                            "type": "LayerConfig",
                            "generator": "ring",
                            "mode": "difference",
                            "location": [0, 0, 0],
                            "filters": [],
                            "args": [256, 64, "l"]
                        }
                    ],
                    "filters": [
                        {
                            "filter": "contrast",
                            "args": []
                        }
                    ],
                    "color": [
                        "hsl(200, 100%, 75%)",
                        "hsl(200, 100%, 25%)"
                    ],
                    "mode": "difference"            
                }
            ],
            "filters": [
                {
                    "filter": "blur",
                    "args": []
                }
            ],
            "color": [
                "hsl(200, 100%, 75%)",
                "hsl(200, 100%, 25%)"
            ],
            "mode": "difference"            
        },
        "SaveConfig": {
            "filename": "test.jpg",
            "format": "JPEG",
            "mode": "RGB",
            "framerate": 12
        }
    }
    ```

OK, great, but what's the difference there? The differences are:

1.  ImageConfig takes a dict rather than a list.
2.  ImageConfig.layers can either be a LayerConfig or an ImageConfig 
    object.

How about CLI config? Well, I'm not really sure how to nest different 
images in the CLI, so I think this will be something that requires 
manipulating the JSON to do. I can maybe look at allowing JSON config 
to be passed in as a string to the CLI to allow this to be able to be 
scripted without the need for storing the config in a file.


2: CACHING NOISE GENERATION
###########################
Requirement 2 can be done in a couple of ways:

A.  Store the output of the generator in a common location outside 
    of the object, and always check that location before generating 
    new data.
B.  Rather than each layer being a unique instance of the generator 
    object, the layers are copies of the same object, and the result 
    is cashed after the first time it is generated.
C.  The output of the generator is cached in the class rather than 
    in the object, so all instances will return the same value.

A seems inelegant.

B is much more elegant, but it requires me to be able to be able to 
distinguish between when I want to create a new instance and when you 
want to reuse an existing instance. That can probably be done by 
naming layers when I want to reuse them. I would just need to keep a 
library around when I'm creating the layers, so I can look up the 
ones I want to reuse easily.

C avoids having to keep a library of objects, so it's probably more 
elegant. However, it runs into problems if I want to reuse two 
difference instances of the same class. I could maybe address that 
with a lookup table in the class. ValueGenerator.fill would then 
have an optional key parameter that could be used to look up the 
specific result.

It seems then the main differences between B and C are:

*   Whether the library is stored outside or inside the class,
*   Whether the library stores the image or the object.

I think I'm leaning towards C. It's probably a slightly more complex 
solution, but it doesn't require much alteration of how the generator 
objects are instantiated. I just need to pass a key as part of the 
initialization of the object, rather than having to either override 
object initialization or store a library of objects outside the class. 


3: GENRATORS THAT DUPLICATE IMAGES
##################################
I think I want to implement the other two requirements before I tackle 
the planning for this one.


MULTI-IMAGE OPERATIONS, TAKE 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The discussion in the previous section runs into a key problem: it 
doesn't address how multicolor images would happen. This is a key 
problem since the initial usecase for layer grouping was to allow 
colorization to be applied to different groups in ways that allowed 
for multicolored images. It seems pretty clear that the multi-image 
solution is too fragile, and the image creation image needs to be 
rethought from the ground up. Hurray.


LAYER GROUPING
~~~~~~~~~~~~~~
The key requirements for layer grouping are:

1.  Allow blending operations to select which layers they apply to.
2.  Allow the masking of blending operations.
3.  Allow colorization to occur on layer groups.
4.  Don't force layer groups to colorize.

That last requirement may mean I need two types of layer groups:

*   Grayscale
*   Colorized

