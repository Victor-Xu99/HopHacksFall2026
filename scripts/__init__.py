"""Command-line entry points.

This file is load-bearing: an unrelated `scripts` package exists in site-packages
on some machines, and without `__init__.py` here that one wins the import and
`python -m scripts.<name>` fails.
"""
