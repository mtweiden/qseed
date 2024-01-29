from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit


class PrintDataPass(BasePass):

    async def run(self, circuit: Circuit, data: PassData) -> None:
        print([_ for _ in data.keys()])
