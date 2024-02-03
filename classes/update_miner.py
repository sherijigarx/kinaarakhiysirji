# In your classes directory, create a new Python file or add to an existing one

from classes.aimodel import AIModelService
import asyncio
import bittensor as bt

class UpdateOutdatedMinersService(AIModelService):
    def __init__(self):
        super().__init__()  # Initializes base class components

    async def run_async(self):
        while True:
            await asyncio.sleep(300)  # Wait for 30 minutes
            try:
                await self.filtered_UIDs_Miner()
                bt.logging.info(f"Updated outdated miners: {self.runs_data}")
            except Exception as e:
                bt.logging.error(f"Error during update of outdated miners: {e}")
