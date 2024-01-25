from typing import Any
from typing import Callable
from typing import Sequence

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.runtime import get_runtime

_logger = logging.getLogger(__name__)


class DeleteFromDataPass(BasePass):

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Run DeleteFromDataPass."""
        del data['ForEachBlockPass_specific_pass_down_seed_circuits']
