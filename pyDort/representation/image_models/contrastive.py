from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from clort.clearn.data import ArgoverseObjectDataFrame
from torchvision.models import *
from torchvision.models._api import WeightsEnum

from ..simple import SimpleArgoverseDetectionRepresentation
