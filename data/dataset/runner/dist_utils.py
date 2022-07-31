from torch import distributed as dist
from typing import Callable, List, Optional, Tuple

def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

