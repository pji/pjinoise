========
pjinoise
========

Noise image generation for those not wanting to do it in Photoshop.


How do I run the code?
----------------------
The best ways to get started is to clone this repository to your
local system and take a look at the following files in the root of
the repository:

*   mazer.py: An example script that creates the image of a maze.
*   template.py: A generic template for a script to create an image
    with pjinoise.

I'll be trying to add more documentation as I go, but this really
wasn't intended to be used by anyone other than me. I can't promise
it will ever be fully documented.


Why can't I install the required statuswriter package from pip?
---------------------------------------------------------------
You can, but it's not up in PyPI yet. To install it:

*   Clone the statuswriter repository from https://github.com/pji
    /statuswriter
*   Run the following from pip: `pip install <path/to/statuswriter>`

Replace "<path/to/statuswriter>" with the path to your clone of the
the statuswriter repository.

NOTE: statuswriter requires Python 3.9.


Can I install this as a package from pip?
-----------------------------------------
Not at this point. If someone else ever reads this and would like
that, let me know. I do plan on doing it eventually just to learn
how to do it. I'm not in any rush, though.


How do I run the tests?
-----------------------
The unit tests for the pjinoise module are built using the standard 
Python unittest module. You can run them after cloning the repository 
to your local system with the following command::

    python3 -m unittest discover tests

There is a shortcut script to run the unit tests, style checks, and
remove trailing whitespace::

    python3 precommit.py


How do I contribute?
--------------------
At this time, this is code is really just me exploring and learning.
I've made it available in case it helps anyone else, but I'm not really
intending to turn this into anything other than a personal project.

That said, if other people do find it useful and start using it, I'll
reconsider. If you do use it and see something you want changed or
added, go ahead and open an issue. If anyone ever does that, I'll
figure out how to handle it.