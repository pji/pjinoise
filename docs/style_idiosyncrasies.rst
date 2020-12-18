====================
Style Idiosyncrasies
====================

My goal is to follow PEP-8 as much as seems reasonable. However, I am 
the only person writing this and I'm using BBEdit rather than a 
dedicated Python IDE like PyCharm, so there may be a few weird things. 

1.  Argument expansion of dicts built within calls
2.  Working tests in __main__ checks
3.  Color values as hexadecimal integers


Argument Expansion of Dicts Built within Calls
----------------------------------------------
This looks like::

    exp = m.Layer(**{
        'source': s.Spot(**{
            'radius': 128,
            'ease': 'l',
        }),
        'blend': op.difference,
    })

That's weird. Why in the hell am I doing it?

The main reason is that BBEdit doesn't fold function calls. However, 
it will fold the dictionary definition within the function call. So, 
defining the dictionary in the call and passing it as keyword 
arguments allows me to fold calls with several parameters.


Working Tests in __main__ Checks
--------------------------------
Sometimes I don't know exactly how code is going to work before I 
implement it. In those cases, I'll often have some code after a 
__main__ check in the module to allow me to experiment with it. The 
main places this shows up in pjinoise are:

*   ease.py
*   filters.py
*   sources.py
*   operations.py

These do math with arrays, and while I have a sense of what the output 
should be I don't know it well enough to figure out the expected value 
of a unit test. Rather than guessing, I'll tend to use the code after 
the __main__ check to build those expected values for me.

I don't know that I'll get rid of this fully, but some of it does need 
to be cleaned up and turned into common utility functions rather than 
loose code sitting at the bottom of modules.


Color Values as Hexadecimal Integers
------------------------------------
The pjinoise module tries, as much as possible, to work with color
values as floats in the range of 0 <= x <= 1. The reason for this is
some blending operations and easing functions only work with data in
that range, so it seemed more efficient to keep the data as floats
rather than having to convert it from 8-bit unsigned integers to
floats and back to 8-bit unsigned integers every time one of those
operations or functions was used.

However, an array of 8-bit integers sure looks a lot cleaner than an
array of floats when written out as test data. So, in most cases, test
data will present arrays of image data as array of hexadecimal integers
rather than an array of floats, and then will convert the data to
floats. This looks like::

    a = np.array([
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
    ], dtype=float)
    a = a / 0xff

It's unlikely this will ever change.
