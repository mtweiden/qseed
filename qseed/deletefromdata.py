from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit


class DeleteSeedsPass(BasePass):

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if 'seed_circuits' in data:
            del data['seed_circuits']
