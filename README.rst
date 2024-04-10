.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/


.. image:: https://img.shields.io/pypi/v/mexican_sign_language_toolkit.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/mexican_sign_language_toolkit/



|

=============================
Gesture spatial language toolkit
=============================


    Scientific computing toolkit designed to support visual languages such as mexican sign detection and analysis.


Installation from pypi
======================

.. code-block:: bash

   pip install mexican_sign_language_toolkit


CLI
============


Detect segment in image
+++++++++++++++++++++++

.. code-block:: bash

   msl -lv [file_path]


Launch web server
+++++++++++++++++++++++

.. code-block:: bash

   msl -s


Create language space
+++++++++++++++++++++++

.. code-block:: bash

   msl -cs

The command in question is designed to search and organize image files in a way that facilitates the creation of a sequenced dataset, particularly useful for image recognition or gesture recognition tasks. It looks through the current directory for all images with .png or .jpg extensions. The identified images are then utilized to form a sequence that corresponds to the filenames, allowing for the creation of examples that share the file's base name. Additionally, the command generates two .npy (NumPy array) files, which are likely used to store the organized data for further processing.

Let's break down the provided example for clarity:

1. You have a dataset related to the word "rabbit," which includes images associated with two specific gestures named "rabbit-1" and "rabbit-2."

2. To improve the recognition accuracy, you need multiple examples of these gestures. Therefore, you have images named "rabbit-1. example1.jpg" and "rabbit-1.example2.jpg" as variations or examples of the "rabbit-1" gesture, and a single image "rabbit-2.jpg" for the second gesture.

3. The command organizes these images in a sequence that reflects their naming convention.

4. If a gesture recognition system is trained with this sequenced data, it would recognize the sequence of gestures "rabbit-1" followed by "rabbit-2" and interpret them as the word "rabbit."


Python
==========

Consult our notebooks to learn more. We recommend you starting with Getting started.

Docs
https://carlos-eduardo-sanchez-torres.sanchezcarlosjr.com/Lenguajes-gesto-espaciales-375c333cdcfb41ef9bdc2cc97146e1b6

Use cases
==========
* Training rutine checkers.
* Video reactions on MacBook Pro.
* Sign languages.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.4. For details and usage
information on PyScaffold see https://pyscaffold.org/.
