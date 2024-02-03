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
        self.get_latest_commit()
        self.vcdnp = self.config.vcdnp
        self.max_mse = self.config.max_mse
        if AIModelService._scores is None:
            AIModelService._scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
        self.scores = AIModelService._scores
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.outdated_miners_set = []
        self.runs_data = []
        self.runs_data_valid = []
        # Set up wandb API
        # Set up wandb API
        self.api = wandb.Api()
        # Define the project path
        self.project_path = "subnet16team/AudioSubnet_Miner"
        self.project_path_valid = "subnet16team/AudioSubnet_Valid"
        # List all runs in the project
        self.latest_commit = self.get_latest_commit()
        self.runs = self.api.runs(self.project_path)
        self.runs_valid = self.api.runs(self.project_path_valid)
        self.runs_miner = self.api.runs(self.project_path)
        # Directory where we will download the metadata files
        self.download_dir = "./"
        self.download_dir_valid = "./neurons"
        # self.filtered_UIDs()
        # self._semaphore = asyncio.Semaphore(10)
        # loop = asyncio.get_event_loop()
        # self.outdated_miners_set = loop.run_until_complete(self.filtered_UIDs())

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

    def get_latest_commit(self):
        owner = "UncleTensor"
        repo = "AudioSubnet"
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        response = requests.get(url)

        if response.status_code == 200:
            commits = response.json()
            return commits[0]['sha'] if commits else None
        else:
            bt.logging.error(f"Failed to fetch the latest commit: {response.status_code}")
            return None
            
    async def filtered_UIDs_valid(self):
        # Get the latest commit SHA
        self.runs_data_valid = []

        for run in self.runs_valid:
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
                    file.download(root=self.download_dir_valid, replace=True)
                    file_path = os.path.join(self.download_dir_valid, file.name)
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                        if 'git' in metadata:
                            run_data['Git Commit'] = metadata['git']['commit']

            # Filter out runs not having the latest commit hash
            if run_data['Git Commit'] == self.latest_commit:
                await self.runs_data_valid.append(run_data['UID'])
                self.runs_data_valid = list(set(self.runs_data_valid))

    async def filtered_UIDs_Miner(self):
        self.runs_data = []

        for run in self.runs_miner:
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
            if run_data['Git Commit'] != self.latest_commit:
                self.runs_data.append(run_data['UID'])
                self.runs_data = list(set(self.runs_data))

    # async def download_and_check_file(self, file, download_dir, latest_commit):
    #     # This function now properly awaits the coroutine for downloading
    #     async def download_file():
    #         # Simulate an asynchronous file download operation
    #         await asyncio.sleep(1)  # Placeholder for actual async download logic
    #         try:
    #             file_path = f"{download_dir}/{file.name}"  # Simulated file path
    #             # You should replace the above lines with the actual download logic
    #             bt.logging.info(f"Downloading file {file.name} to {file_path}")
    #             return file_path
    #         except Exception as e:
    #             bt.logging(f"Error during file download: {e}")
    #             return None

    #     async def check_file_commit(file_path):
    #         async with aiofiles.open(file_path, 'r') as f:
    #             metadata = json.loads(await f.read())
    #             git_commit = metadata.get('git', {}).get('commit', None)
    #             return git_commit == latest_commit

    #     # Run the blocking file download in a thread pool only if necessary
    #     # If your download operation is already asynchronous, just await it directly
    #     file_path = await download_file()
    #     # Once downloaded, check the commit asynchronously
    #     return await check_file_commit(file_path)

    # async def process_run(self, run, latest_commit):
    #     tasks = []
    #     for file in run.files():
    #         if file.name == 'wandb-metadata.json':
    #             task = asyncio.create_task(self.download_and_check_file(file, self.download_dir, latest_commit))
    #             tasks.append(task)
    #     results = await asyncio.gather(*tasks)
    #     # Assuming `download_and_check_file` returns True if the commit matches the latest commit
    #     if any(results):
    #         bt.logging.info(f"Run {run.config['uid']} is using the latest commit.")
    #     else:
    #         self.runs_data.append(run.config['uid'])
    #         bt.logging.info(f"Run {run.config['uid']} has outdated commit.")



    # # async def periodically_update_outdated_miners(self):
    # #     while True:
    # #         bt.logging.info("Starting to update outdated miners.")
    # #         await asyncio.sleep(300)  # 30 minutes
    # #         try:
    # #             self.outdated_miners_set = await self.filtered_UIDs()
    # #             bt.logging.info(f"Updated outdated miners: {self.outdated_miners_set}")
    # #         except Exception as e:
    # #             bt.logging.error(f"Error during update of outdated miners: {e}")
    # #         bt.logging.info("Completed updating outdated miners.")
    # # Adjust the number based on your needs

    # async def fetch_and_process_runs(self, latest_commit):
    #     tasks = []
    #     bt.logging.info(f"All RUNS.........................................: {self.runs}")
    #     for run in self.runs:
    #         if run.state == 'running':
    #             bt.logging.info(f" run state is ............................................................... {run.state}")
    #             async with self._semaphore:
    #                 task = asyncio.create_task(self.process_run(run, latest_commit))
    #                 tasks.append(task)
    #     await asyncio.gather(*tasks)


    # async def filtered_UIDs(self):
    #     latest_commit = await self.get_latest_commit("UncleTensor", "AudioSubnet")
    #     bt.logging.info(f"Latest commit.........................................: {latest_commit}")
    #     await self.fetch_and_process_runs(latest_commit)
    #     self.runs_data = list(set(self.runs_data))  # Deduplicating the UIDs
    #     return self.runs_data
