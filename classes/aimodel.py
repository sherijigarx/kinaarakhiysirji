import os
import argparse
import bittensor as bt
import sys
import asyncio
import traceback
from datasets import load_dataset
import torch
import random
import csv
import pandas as pd
import torchaudio
from tabulate import tabulate
# Import your module
import lib.utils
import lib
import traceback
import platform
import psutil
import GPUtil
import wandb
import requests
import json
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiohttp
import json

class AIModelService:
    _scores = None

    def __init__(self):
        self.config = self.get_config()
        self.sys_info = self.get_system_info()
        self.setup_paths()
        self.setup_logging()
        self.setup_wallet()
        self.setup_subtensor()
        self.setup_dendrite()
        self.setup_metagraph()
        self.vcdnp = self.config.vcdnp
        self.max_mse = self.config.max_mse
        if AIModelService._scores is None:
            AIModelService._scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
        self.scores = AIModelService._scores
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.outdated_miners_set = []
        self.runs_data = []
        # Set up wandb API
        self.api = wandb.Api()
        # Define the project path
        self.project_path = "subnet16team/AudioSubnet_Miner"
        # List all runs in the project
        self.runs = self.api.runs(self.project_path)
        # Directory where we will download the metadata files
        self.download_dir = "./"
        # self.filtered_UIDs()
        loop = asyncio.get_event_loop()
        self.outdated_miners_set = loop.run_until_complete(self.filtered_UIDs())

    def get_config(self):
        parser = argparse.ArgumentParser()

        # Add arguments as per your original script
        parser.add_argument("--alpha", default=0.9, type=float, help="The weight moving average scoring.")
        parser.add_argument("--custom", default="my_custom_value", help="Adds a custom value to the parser.")
        parser.add_argument("--netuid", type=int, default=16, help="The chain subnet uid.")
        parser.add_argument("--vcdnp", type=int, default=10, help="Number of miners to query for each forward call.")
        parser.add_argument("--max_mse", type=float, default=1000.0, help="Maximum Mean Squared Error for Voice cloning.")

        # Add Bittensor specific arguments
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)

        # Parse and return the config
        config = bt.config(parser)
        return config

    def get_system_info(self):
        system_info = {
            "OS -v": platform.platform(),
            "CPU ": os.cpu_count(),
            "RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB", 
        }

        gpus = GPUtil.getGPUs()
        if gpus:
            system_info["GPU"] = gpus[0].name 

        # Convert dictionary to list of strings
        tags = [f"{key}: {value}" for key, value in system_info.items()]
        tags.append(lib.__version__)
        return tags

    def setup_paths(self):
        # Set the project root path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Set the 'AudioSubnet' directory path
        audio_subnet_path = os.path.abspath(project_root)

        # Add the project root and 'AudioSubnet' directories to sys.path
        sys.path.insert(0, project_root)
        sys.path.insert(0, audio_subnet_path)

    def setup_logging(self):
        # Set up logging with the provided configuration and directory
        self.config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                self.config.logging.logging_dir,
                self.config.wallet.name,
                self.config.wallet.hotkey,
                self.config.netuid,
                "validator",
            )
        )
        
        # Ensure the logging directory exists
        if not os.path.exists(self.config.full_path):
            os.makedirs(self.config.full_path, exist_ok=True)

        bt.logging(self.config, logging_dir=self.config.full_path)

    def setup_wallet(self):
        # Initialize the wallet with the provided configuration
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")


    def setup_subtensor(self):
    # Initialize the subtensor connection with the provided configuration
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

    def setup_dendrite(self):
        # Initialize the dendrite (RPC client) with the wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

    def setup_metagraph(self):
        # Initialize the metagraph for the network state
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

    def update_score(self, axon, new_score, service, ax):
            try:
                uids = self.metagraph.uids.tolist()
                zipped_uids = list(zip(uids, self.metagraph.axons))
                uid_index = list(zip(*filter(lambda x: x[1] == axon, zipped_uids)))[0][0]
                if uid_index in ax:
                    alpha = self.config.alpha
                    self.scores[uid_index] = alpha * self.scores[uid_index] * (1 - alpha) * new_score * 0.0

                else:
                    alpha = self.config.alpha
                    self.scores[uid_index] = alpha * self.scores[uid_index] + (1 - alpha) * new_score
                    bt.logging.info(f"Updated score for {service} Hotkey {axon.hotkey}: {self.scores[uid_index]}")
            except Exception as e:
                print(f"An error occurred while updating the score: {e}")


    def punish(self, axon, service, punish_message):
        '''Punish the axon for returning an invalid response'''
        try:
            uids = self.metagraph.uids.tolist()
            zipped_uids = list(zip(uids, self.metagraph.axons))
            uid_index = list(zip(*filter(lambda x: x[1] == axon, zipped_uids)))[0][0]
            alpha = self.config.alpha
            self.scores[uid_index] = alpha * self.scores[uid_index] + (1 - alpha) * (-1)
            if self.scores[uid_index] < 0:
                self.scores[uid_index] = 0
            # Log the updated score
            bt.logging.info(f"Score after punishment for Hotkey {axon.hotkey} using {service} is Punished  Due to {punish_message} : {self.scores[uid_index]}")
        except Exception as e:
            print(f"An error occurred while punishing the axon: {e}")

    async def run_async(self):
        raise NotImplementedError
    
    def get_config_value(self, config, key):
        """Safely extract the value from the config, accommodating different data structures."""
        value = config.get(key)
        if isinstance(value, (int, str)):
            return value
        elif isinstance(value, dict):
            return value.get('value', 'N/A')
        return 'N/A'



    async def get_latest_commit(self, owner, repo):
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    commits = await response.json()
                    return commits[0]['sha'] if commits else None
                else:
                    return None

    async def filtered_UIDs(self):
        owner = "UncleTensor"  # Replace with actual GitHub owner
        repo = "AudioSubnet"    # Replace with actual GitHub repository

        # Get the latest commit SHA
        latest_commit = await self.get_latest_commit(owner, repo)
        self.runs_data = []

        for run in self.runs:
            if run.state != 'running':
                continue

            # Initialize data dictionary for this run
            run_data = {
                'UID': self.get_config_value(run.config, 'uid'),
                'Hotkey': self.get_config_value(run.config, 'hotkey'),
                'Git Commit': 'null'
            }

            files = run.files()
            for file in files:
                if file.name == 'wandb-metadata.json':
                    file.download(root=self.download_dir, replace=True)
                    file_path = os.path.join(self.download_dir, file.name)
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                        if 'git' in metadata:
                            run_data['Git Commit'] = metadata['git']['commit']

            # Filter out runs not having the latest commit hash
            if run_data['Git Commit'] != latest_commit:
                await self.runs_data.append(run_data['UID'])
                self.runs_data = list(set(self.runs_data))

    # async def process_run(self, run, latest_commit):
    #     # Assuming 'run' is an object with a method 'files()' that returns a list of file objects.
    #     # Each file object needs a synchronous 'download()' method call.
    #     async def download_and_check_file(file, download_dir):
    #         # Synchronous code to download the file and check its commit
    #         file_path = file.download(root=download_dir, replace=True)  # Assuming this returns the path
    #         with open(file_path, 'r') as f:
    #             metadata = json.load(f)
    #             git_commit = metadata['git']['commit'] if 'git' in metadata else None
    #             bt.logging.info(f"Run {run} and  {run.config['uid']} has git commit................................: {git_commit}")

    #             return git_commit == latest_commit
        
    #     with ThreadPoolExecutor() as pool:
    #         # Offload the blocking operation to a separate thread
    #         results = await asyncio.gather(*[
    #             asyncio.to_thread(download_and_check_file, file, self.download_dir)
    #             for file in run.files() if file.name == 'wandb-metadata.json'
    #         ])
    #         self.runs_data = []
    #         # Process the results, which indicate whether the run uses the latest commit
    #         if any(results):
    #             pass
    #         else:
    #             # No files match the latest commit, consider this run as outdated
    #             bt.logging.info(f"The UID with not the latest commit is: {run.config['uid']}")
    #             self.runs_data.append(run.config['uid'])
    #             self.runs_data = list(set(self.runs_data))

    async def download_and_check_file(self, file, download_dir, latest_commit):
    # Asynchronously download the file and check its commit
    # Assuming you have a way to download files asynchronously,
    # for demonstration, replace the following with your actual download logic
        async with aiohttp.ClientSession() as session:
            async with session.get(file.download_url) as resp:  # Presuming `file` has a `download_url` attribute
                if resp.status == 200:
                    content = await resp.text()
                    file_path = os.path.join(download_dir, file.name)
                    async with aiofiles.open(file_path, 'w') as f:
                        await f.write(content)
                    # After downloading, check the commit
                    async with aiofiles.open(file_path, 'r') as f:
                        metadata = json.loads(await f.read())
                        git_commit = metadata.get('git', {}).get('commit', None)
                        return git_commit == latest_commit
        return False  # Default return if conditions fail

    async def process_run(self, run, latest_commit):
        tasks = []
        for file in run.files():
            if file.name == 'wandb-metadata.json':
                task = asyncio.create_task(self.download_and_check_file(file, self.download_dir, latest_commit))
                tasks.append(task)
        results = await asyncio.gather(*tasks)
        if not any(results):
            self.runs_data.append(run.config['uid'])            
        


    async def fetch_and_process_runs(self, latest_commit):
        tasks = [asyncio.create_task(self.process_run(run, latest_commit)) for run in self.runs if run.state == 'running']
        bt.logging.info(f"Fetching and processing {len(tasks)} runs")
        await asyncio.gather(*tasks)

    async def filtered_UIDs(self):
        latest_commit = await self.get_latest_commit("UncleTensor", "AudioSubnet")
        await self.fetch_and_process_runs(latest_commit)
        self.runs_data = list(set(self.runs_data))  # Deduplicating the UIDs
        return self.runs_data
