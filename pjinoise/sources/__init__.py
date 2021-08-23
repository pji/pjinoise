"""
sources
~~~~~~~

Objects that generate values, both patterned and noise.


Basic Usage
===========
A Source object is used to create an array of data that can be
used for various things, but the main use case is the creation of
still images and video. Creating an instance of a subclass of
Source (a "source") works like instantiating any other Python
class. The specific parameters vary based on what the specifc source
will create.

Usage::

    >>> src = Lines('h', 22.2, 'ios')
    >>> src
    Lines(direction='h', length=22.2, ease='ios')


fill()
------
The fill() method is used to generate data from a source. The fill()
function always adheres to the following protocol:

    :param size: A sequence containing three integers that give the
        Z, Y, and X dimensions of the data array to produce.
    :param loc: (Optional.) A sequence containing three integers that
        offset the data generation within the total space of data the
        source could generate.

        If you think of each position in the output array as a pixel
        in an image, most sources use the location of that pixel within
        the image to determine what the value of that pixel should be.
        The loc parameter allows you to offset the location of all the
        pixels in the image by the given amount, which will change the
        values of the output.
    :return: A numpy n-dimensional array containing floats within the
        range 0 <= x <= 1.
    :rtype: numpy.ndarray

Usage::

    >>> lines = Lines('h', 22.2, 'ios')
    >>> size = (1, 2, 3)
    >>> location = (2, 3, 4)
    >>> lines.fill(size, location)
    array([[[0.45560205, 0.45560205, 0.45560205],
            [0.60298931, 0.60298931, 0.60298931]]])


asdict()
--------
The asdict() method is primarily used to make serializing source
objects to JSON easier. It also can be used in testing to compare
two source objects. The following is true of the output of asdict():

*   The "type" key contains the serialization name for the object's
    class.
*   The other keys are the parameters for creating a new instance
    of the objects class that is a copy of the object.
*   The values of those parameters are data types that can be
    serialized by the json module.

Note: When creating subclasses of ValuesSource, care should be taken
to ensure the subclass's asdict() method only returns data that can
be serialized by the json module. There are a couple of ways to
accomplish this:

*   Source.asdict() won't return private attributes in the
    output dictionary, so the serialized version can be stored in
    the public attribute (or at least returned by calls to the
    public attribute) and the deserialized version stored in a
    private attribute.
*   The asdict() method can be subclassed to serialize the value of
    the attribute before the output dictionary is returned.

Usage:

    >>> lines = Lines('h', 22.2, 'ios')
    >>> lines.asdict()
    {'direction': 'h', 'length': 22.2, 'type': 'lines', 'ease': 'ios'}


Serialization Helpers
=====================
The sources module has a few capabilities to help with serialization.
The get_regname_for_class() function returns a string that represents
a source class that has been registered with the sources module.

Usage::

    >>> obj = Lines('h', 22.2, 'ios')
    >>> get_regname_for_class(obj)
    'lines'

New source classes can be registered with the source module by adding
them to the registered_source dictionary. The key should be the
short string you want to use as the serialized value of the class.

Usage::

    >>> class Spam(Source):
    ...     def fill(*args):
    ...         pass
    ...
    >>> registered_sources['spam'] = Spam
    >>> obj = Spam()
    >>> get_regname_for_class(obj)
    'spam'

The primary purpose for this feature is to allow classes to be easily
deserialized from JSON objects. It also should provide a measure of
input validation at deserialization to reduce the risk of remote code
execution vulnerabilities though deserialization.
"""
from typing import Dict

from pjinoise.sources.path import AnimatedPath, Path, SolvedPath, TilePaths
from pjinoise.sources.random import *
from pjinoise.sources.source import *
from pjinoise.sources.static import *
from pjinoise.sources.unit import *


# Registration.
registered_sources = {
    'gradient': Gradient,
    'lines': Lines,
    'rays': Rays,
    'ring': Ring,
    'solid': Solid,
    'spheres': Spheres,
    'spot': Spot,
    'text': Text,
    'waves': Waves,

    'curtains': Curtains,
    'cosinecurtains': CosineCurtains,
    'embers': Embers,
    'worley': Worley,
    'worleycell': WorleyCell,
    'path': Path,
    'animatedpath': AnimatedPath,
    'solvedpath': SolvedPath,
    'perlin': Perlin,
    'random': Random,
    'seededrandom': SeededRandom,
    'values': Values,

    'oldoctavecosinecurtains': OldOctaveCosineCurtains,
    'octavecosinecurtains': OctaveCosineCurtains,
    'octaveperlin': OctavePerlin,

    'cachingoctaveperlin': CachingOctavePerlin,

    'tilepaths': TilePaths,
}


# Registration utility functions.
def deserialize_source(attrs: Dict) -> 'Source':
    cls = registered_sources[attrs['type']]
    del attrs['type']
    try:
        return cls(**attrs)
    except TypeError as e:
        msg = f'{cls.__name__} raised following: {e}.'
        raise TypeError(msg)


def get_regname_for_class(obj: object) -> str:
    regnames = {registered_sources[k]: k for k in registered_sources}
    clsname = obj.__class__
    return regnames[clsname]


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    kwargs = {
        'seed': b'\x00\x01\x02\x03',
    }
    cls = SeededRandom
    size = (2, 8, 8)
    obj = cls(**kwargs)
    val = obj.fill(size)
    c.print_array(val)
