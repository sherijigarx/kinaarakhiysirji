# In your classes directory, create a new Python file or add to an existing one

from classes.aimodel import AIModelService
import asyncio

class UpdateOutdatedMinersService(AIModelService):
    def __init__(self):
        super().__init__()  # Initializes base class components

    async def run_async(self):
        while True:
            await asyncio.sleep(300)  # Wait for 30 minutes
            try:
                _ = await self.filtered_UIDs()
                print(f"Updated outdated miners: {self.outdated_miners_set}")  # Or use logging
            except Exception as e:
                print(f"Error during update of outdated miners: {e}")  # Or use logging
