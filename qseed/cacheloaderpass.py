from typing import Any
from typing import Callable
from typing import Sequence

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.runtime import get_runtime

_logger = logging.getLogger(__name__)


class CacheLoaderPass(BasePass):

    def __init__(
        self,
        cache_key: str,
        load_function: Callable,
        load_function_args: Sequence[Any] | None = None,
    ) -> None:
        """
        Construct a CacheLoaderPass.

        This pass checks if `cache_key` is loaded into a worker's cache. If
        not, then it calls `load_function`.

        args:
            cache_key (str): The key of the object to be placed into worker
                cache.

            load_function (Callable): The function which describes how to
                load the desired object so that it can be placed in cache.

            load_function_args (Sequence[Any] | None): Arguments to pass to
                the load_function. (Default: None)
        """
        if not isinstance(cache_key, str):
            raise TypeError(
                f'cache_key must be a str, got {type(cache_key)}.'
            )
        if not isinstance(load_function, Callable):
            raise ValueError('load_function must be a Callable.')
        self.cache_key = cache_key
        self.load_function = load_function
        self.load_function_args = load_function_args

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Run CacheLoaderPass."""
        worker_cache = get_runtime().get_cache()
        if self.cache_key not in worker_cache:
            if self.load_function_args is not None:
                obj = self.load_function(*self.load_function_args)
            else:
                obj = self.load_function()
            worker_cache[self.cache_key] = obj
