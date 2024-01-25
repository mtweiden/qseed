
import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from timeit import default_timer

_logger = logging.getLogger(__name__)


class TimePass(BasePass):

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Run CacheLoaderPass."""
        if 'time' not in data:
            data['time'] = default_timer()
        else:
            duration = default_timer() - data['time']
            print(f'{duration:>0.3f}s')
