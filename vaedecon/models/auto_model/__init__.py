"""Utils class allowing to reload any :class:`pythae.models` automatically with the following
lines of code.

.. code-block::

    >>> from vaedecon.models import AutoModel
    >>> model = AutoModel.load_from_folder(model_dir='path/to/my_model')
"""

from .auto_config import AutoConfig
from .auto_model import AutoModel
