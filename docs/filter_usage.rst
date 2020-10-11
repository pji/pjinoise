============
Filter Usage
============

This document describes how to invoke the built-in filters from the 
command line interface.


The Filter Language
-------------------
An example filter invocation looks like this:

    ```
    rotate90_2:1_r+skew_3:1_10+skew_3:2_-10
    ```

A individual filter invocation has three parts:

    *   Name: the registered name of the filter
    *   Period: how often the filter is used
    *   Parameters: arguments for the filter

Each part of the individual filter invocation is delimited by an 
underscore (_) character. So, if we have the following invocation:

    ```
    rotate90_2:1_r
    ```

It breaks down to the following

    *   The filter name is `rotate90`
    *   The filter period is `2:1`
    *   The filter parameter is `r`

Multiple filters can be chained together, using a plus sign (+) as 
a delimiter. The same filter can be invoked multiple times with 
different periods or parameters.