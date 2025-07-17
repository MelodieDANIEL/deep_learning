.. slide::

Document editing
================

.. slide::

Text formatting
---------------

The documents are written in reStructuredText, you can use:

.. discoverList::
    
    * ``*italic*`` for *italic* text
    * ``**bold**`` for **bold** text
    * ````code```` for ``code``
    * The ``.. center::`` directive to center text

.. slide::

Links and referencing
---------------------

External
~~~~~~~~

For external links, use this syntax: ```link <https://google.com>`_``, result: `link <https://google.com>`_


.. slide::

Emphasis boxes
--------------

.. important::

    This is the result of ``.. important::``

.. note::

    This is the result of ``.. note::``

.. warning::

    This is the result of ``.. warning::``

.. success::

    This is the result of ``.. success::``

.. slide::

Slides and text
---------------

In the document, you can use the ``.. slide::`` directive to create a new slide.

.. discover::

    You can use the ``..discover::`` directive for a zone that will appear progressively.

.. discover::

    The ``..textOnly::`` directive will hide the content in the slides, and the ``..slideOnly::`` directive will hide the content of the document.

.. slide::

Math
----

You can use embedded *LaTeX* $$e^{i\pi} = -1$$.

.. discover::

    Use the ``math`` directive for display math:

    .. math::

        \sum_{i=1}^{n} i = \frac{n(n+1)}{2}

.. slide::

Code
----

The ``..code-block:: language`` directive can be used to display code

.. code-block:: python

    # A simple function
    def hello():
        print("Hello, world!")

.. slide::

Spoiler
-------

You can hide content with the ``.. spoiler::`` directive.

.. spoiler::

    This is a spoiler!

.. slide::

Images
------

The ``..image::`` directive can be used to display images:

.. center::
    
        .. image:: images/html5.jpg
            :width: 300

.. slide::

YouTube
-------

The ``..youtube::`` directive can be used to embed YouTube videos:

.. center::
    .. youtube:: dQw4w9WgXcQ
