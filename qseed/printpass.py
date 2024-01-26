
import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from timeit import default_timer

_logger = logging.getLogger(__name__)


class PrintPass(BasePass):

    def __init__(self, message: str) -> None:
        self.msg = message

    async def run(self, circuit: Circuit, data: PassData) -> None:
        print(self.msg)
