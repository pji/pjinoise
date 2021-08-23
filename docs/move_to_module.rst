*******************
Moving to a Package
*******************

This document creates a plan for implementing pjinoise as a package for
importing into applications like mazer.


Why a package?
==============
The main purpose of this is to clean up mazers. It's a fairly well
defined script with a single function. It would be useful to have it
as a separate thing that uses pjinoise rather than it only existing as
an example in the root of the pjinoise repository. Also, this would
allow mazers to use older versions of pjinoise if I'm working on any
breaking changes to pjinoise.


Goals for the package
=====================
My goals for this package are:

#.  Rethink pjinoise as an API rather than a CLI application.
#.  Improve test coverage.
#.  Follow a model like pillow for the API documentation.
#.  Do new work on the dev branch rather than straight to main.


Should the name change?
=======================
What should I do about the name pjinoise? It has a couple of strikes
against it:

#.  It explicitly references my name.
#.  It does more than generate noise.

The first issue isn't that much of a problem. My name is only there as
a way to ensure uniqueness, since I have uncommon initials. Since I'm
probably the only person that will ever use this, I'm not that worried
about the weirdness of my name being part of it.

Still, it's not a common practice in Python package naming, as far as I
can tell. So, maybe it would be better to find something different.

The second issue is a bit more of a concern. pjinoise produces patterns
as well as noise, so it probably makes sense to have a name that better
describes what the package does. Of course, the package is sort of a
kitchen sink of image generationy things, so coming up with a fitting
name that isn't just "PJI Image Maker Thingie" might be tough.


Official guidance
-----------------
Python's guidance around package naming is in PEP423_.

.. _PEP423: https://www.python.org/dev/peps/pep-0423/

It seems to recommend the top level of the name identify the owner of
the project. In that case, pjinoise would become something like 
pji.noise. Details about this namespace packaging can be found in the
PyPA guide_ on namespace packages.

.. _guide: https://packaging.python.org/guides/packaging-namespace-packages/


Subpackage name options
-----------------------
Some options for subpackage names include:

*   noise
*   image
*   pattern
*   proceduralimage
*   ymage
*   imagegen
*   imagemaker
*   patternmaker
*   patternizer
*   imaginer
*   patterndistorter
*   painter
*   visual-pattern-maker
*   imagerender
*   vnoise

I may be running into a problem here because I'm trying to come up with
a name for the whole thing at once. Maybe it does need to be broken up
some more. Though, the core function of generating a layered image
requires all of the pieces involved. So, not sure if there is a good
way to break it up more.

The question of noise or pattern is irrelevant because it can do both.
The name should focus on the core function of creating an image from
generated, interacting layers.

*   layeredimage
*   layervisualizer
*   layers
*   rasterizer
*   imatrix
*   fills
*   texturizer
*   spimage
*   imagecloset
*   stuff
*   raster
*   rstrmkr
*   rastermaker
*   jim
*   imager (used)
*   pjimager
*   rasterframe
*   imgpy (used)
*   rastpy
*   rasty

OK, I like that last one. It hints at what the package does, but it's
not trying to explain it all. So, the name will be pji.rasty, and it
will be the second package in the pji namespace, after statuswriter.


Should any functionality be split out of rasty?
===============================================
If rasty is about the creation of images, is there any functionality
in pjinoise that should be split out? Candidates include:

*   The CLI interface.
*   The IO.

The CLI interface is a fairly obvious drop. I'd already moved to using
the Python template to create images anyway because pjinoise was too
complex for a good CLI. Mazer has its own CLI. So, not only should the
CLI function be split out, it should probably just be dropped entirely.

IO is trickier, since rasty is about creating images. What's the point
of creating an image if you don't save it out to a file? That said,
the IO functions are very separate from the image data generation
functions. Saving image or video data works with any numpy.ndarray of
the right format. So it feels like a separate package that writes
arrays to image or video files could make sense. rasty could still
allow for the creation of the image files, but it could use the
new pji.imagewriter module for it, rather than having an IO module
of its own.

By that logic, though, any piece that operates on an ndarray can be
pulled out as its own module. That ends up being a large chunks of
pjinoise:

*   pjinoise.ease
*   pjinoise.filter
*   pjinoise.operations

That would leave rasty as just the sources, the layers, and the engine
that puts them together.

Honestly, that may simplify things. Right now, layers have a pretty
awkward structure because I'm thinking about them as layers. But,
maybe that can be simplified if everything is just seen as an ndarray
of image data. Rather than having a layer that has a source with
filters and et cetera, you just have the sources and you apply the
eases, filters, and blends to them. Would that work?

The only real problem with it would be the filter processing. But, I'm
not sure the filter processing is completely necessary. At least, the
filter classes can still have the pre and post processing functionality
but I don't know if I need an engine to run it. That said, the core
functions of that engine are in the pjinoise.filters module anyway, so
I wouldn't be missing much from pjinoise.pjinoise.

So, we're pretty close to a full rewrite here. Maybe we should just go
that route?