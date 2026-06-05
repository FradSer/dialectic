"""Test session setup.

Importing ``dialectic`` loads the package's ``.env`` (via its
``__init__``) before tests are collected, so the ``e2e`` skip guard can see
``GOOGLE_API_KEY`` whether it comes from the real environment or ``.env``.

Mock helpers live in ``tests/helpers.py``.
"""

import dialectic  # noqa: F401  (side effect: loads .env)
