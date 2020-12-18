==========
CLI Design
==========

The purpose of this document is to determine the design of the CLI
for the pjinoise module. The functionality of the module has become
so complex that it's difficult to ensure every option is easily
available from the CLI. This design will look at what functionality
should be available and how it can be made available to a useable CLI.


Purpose
-------
The purposes of pjinoise's CLI are to:

*   Allow image creation to be scripted through shell commands.
*   Allow simple demonstrations of the module's capabilities.