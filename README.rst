========
pjinoise
========

Noise image generation for those not wanting to do it in Photoshop.


How do I run the code?
----------------------
To run pjinoise, clone this repository to your local system and run the 
following from the repository::

    python3 pjinoise -h

That will get you the help information for pjinoise.


How do I run the tests?
-----------------------
The unit tests for the pjinoise module are built using the standard 
Python unittest module. You can run them after cloning the repository 
to your local system with the following command::

    python3 -m unittest discover tests

There is a shortcut script to run the unit tests, style checks, and
remove trailing whitespace::

    python3 precommit.py
