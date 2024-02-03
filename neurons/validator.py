# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): ETG Team
# Copyright © 2023 <ETG>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Base Class
import os
import sys
import asyncio

# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)

# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)
from classes.tts import TextToSpeechService 
from classes.vc import VoiceCloningService



# Update validator.py to include the new UpdateOutdatedMinersService

import asyncio
from classes.tts import TextToSpeechService
from classes.vc import VoiceCloningService
from classes.update_miner import UpdateOutdatedMinersService

async def main():
    tts_service = TextToSpeechService()
    tts_service.new_wandb_run()

    vc_service = VoiceCloningService()
    vc_service.new_wandb_run()

    # update_miners_service = UpdateOutdatedMinersService()

    # Start all services asynchronously
    vc_task = asyncio.create_task(vc_service.run_async())
    tts_task = asyncio.create_task(tts_service.run_async())
    # update_miners_task = asyncio.create_task(update_miners_service.run_async())

    # Wait for all tasks to complete
    await asyncio.gather(vc_task, tts_task)

if __name__ == "__main__":
    asyncio.run(main())

