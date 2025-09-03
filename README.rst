demregpy
========

This is a python implementation of Hannah & Kontar (2012)'s regularized inversion method.
The code is tightly based on the IDL mapping version of the DEM reg-inv code found at https://github.com/ianan/demreg in addition, the code enforces a positivity constraint on the DEM (hence pos).

The philosophy was to produce as similar a piece of software as the original version and as such, this python version has been shown to recover the same DEM as the IDL version (to within approximately 4 significant figures).
It is likely this philosophy has lead to performance hits and I plan to go back and address the more hacky parts of the code at a later date.

To calculate a DEM you first need:

- Data and associated error for a range of channels: e.g. in dn/s/px or counts/s

- Temperature dependent channel response: How sensitive are your channels to plasma of each temperature?

To use: simply call ``dn2dem_pos`` with either a single pixel, 1d slice or 2d map of DN values as a function of filter (and associated error on DN), an array of temperatures over which to perform the DEM analysis and a temperature response for those filters.

For large datasets, beyond 200 pixels, the code switches to a parallel execution providing significant speedups.

Usage of Generative AI
----------------------

We expect authentic engagement in our community.
Be wary of posting output from Large Language Models or similar generative AI as comments on GitHub or any other platform, as such comments tend to be formulaic and low quality content.
If you use generative AI tools as an aid in developing code or documentation changes, ensure that you fully understand the proposed changes and can explain why they are the correct approach and an improvement to the current state.

License
-------

This project is Copyright (c) def and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Openastronomy packaging guide <https://github.com/OpenAstronomy/packaging-guide>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.

Contributing
------------

We love contributions! demregpy is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
demregpy based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
