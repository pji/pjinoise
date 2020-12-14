====================
Style Idiosyncrasies
====================

My goal is to follow PEP-8 as much as seems reasonable. However, I am 
the only person writing this and I'm using BBEdit rather than a 
dedicated Python IDE like PyCharm, so there may be a few weird things. 

1.  Argument expansion of dicts built in calls
2.  Extraneous __main__ checks
3.  Working tests in __main__ checks
4.  Color values as hexadecimal integers


Argument Expansion of Dicts Built in Calls
------------------------------------------
This looks like:

    ```
    exp = m.Layer(**{
        'source': s.Spot(**{
            'radius': 128,
            'ease': 'l',
        }),
        'blend': op.difference,
    })
    ```

That's weird. Why in the hell am I doing it?

The main reason is that BBEdit doesn't fold function calls. However, 
it will fold the dictionary definition within the function call. So, 
defining the dictionary in the call and passing it as keyword 
arguments allows me to fold calls with several parameters.


Extraneous __main__ Checks
--------------------------
These look like this:

    ```
    if __name__ == '__main__':
        raise NotImplementedError
    ```

These were created due to a bug in BBEdit that prevented it from 
folding the last block in Python files. Barebones Software has since 
fixed the bug, and I'll be removing these going forward.


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
This is mainly because color values in files tend to be stored as 
eight or higher bit numbers, and I like columns of numbers to line 
up neatly where possible. 