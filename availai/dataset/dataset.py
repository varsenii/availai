import os
import wandb
from typing import Optional, Dict


class Dataset:
    def __init__(self, path: str, name: Optional[str] = None, project_name: Optional[str] = None,
                 wandb_api_key: Optional[str] = None):
        """
        Initializes a Dataset object. Checks if the dataset path exists and sets the dataset name.

        Args:
            path (str): Path to the dataset directory.
            name (Optional[str]): Name of the dataset. If not provided, it is derived from the directory name.
            project_name (Optional[str]): The name of the Weights & Biases project.
            wandb_api_key (Optional[str]): API key for Weights & Biases. Can be a string (API key) or a path to a file containing the key.

        Raises:
            FileNotFoundError: If the specified dataset path does not exist.
            ValueError: If no valid W&B API key is provided.
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"The specified path '{path}' does not exist or is not a directory.")

        self.path = path
        self.name = name if name else os.path.basename(os.path.normpath(path))
        self.project_name = project_name

        # Authenticate with W&B using the provided API key or file
        self._authenticate(wandb_api_key)

    def create_wandb_artifact(self, metadata: Optional[Dict] = None):
        """
        Creates and logs a dataset artifact to W&B.

        Args:
            metadata (Optional[Dict]): Metadata to be attached to the artifact. Default is None.

        Raises:
            Exception: If there is an error during artifact creation.
        """
        try:
            with wandb.init(project=self.project_name, job_type='upload_data') as run:
                artifact = wandb.Artifact(name=self.name, type='dataset', metadata=metadata or {})
                artifact.add_dir(self.path)
                run.log_artifact(artifact)
                print(f"Artifact '{self.name}' created and logged successfully.")
        except Exception as e:
            print(f"Error creating artifact: {e}")

    def log_table_on_wandb(self):
        try:
            with wandb.init(project=self.project_name, job_type='create_table'):
                table = wandb.Table(columns=['a', 'b'], data=[['a1', 'b1'], ['a2', 'b2']])
                wandb.log({'Football player dataset': table})
        except Exception as e:
            print(f'Error creating table: {e}')

    def fetch_wandb_artifact(self, artifact_name: str):
        """
        Fetches an artifact from W&B and downloads it to a local directory.

        Args:
            artifact_name (str): The name of the artifact to fetch from W&B.

        Returns:
            str: Path to the downloaded artifact directory.

        Raises:
            Exception: If there is an error during artifact fetching.
        """
        try:
            with wandb.init(project=self.project_name, job_type='load_data') as run:
                artifact = run.use_artifact(artifact_name, type='dataset')
                artifact_dir = artifact.download()
                print(f"Artifact '{artifact_name}' fetched and downloaded to: {artifact_dir}")
                return artifact_dir
        except Exception as e:
            print(f"Error fetching artifact '{artifact_name}': {e}")
            return None

    def preprocess(self, target: str, steps: Dict, metadata: Optional[Dict] = None):
        """
        Preprocesses the dataset and logs the new dataset artifact on W&B.

        Args:
            target (str): The target operation for preprocessing (e.g., resize, normalize).
            steps (Dict): A dictionary defining the preprocessing steps.
            metadata (Optional[Dict]): Metadata for the new artifact. Default is None.

        Raises:
            Exception: If there is an error during preprocessing or artifact creation.
        """
        try:
            print(f"Preprocessing dataset with target '{target}' and steps: {steps}")
            metadata = metadata or {}
            metadata['preprocessing_steps'] = steps
            self.create_wandb_artifact(metadata=metadata)
        except Exception as e:
            print(f"Error during preprocessing: {e}")

    def _authenticate(self, wandb_api_key: Optional[str]):
        """
        Authenticates with Weights & Biases using the provided API key or file path.

        Args:
            wandb_api_key (Optional[str]): API key or file path for Weights & Biases authentication.

        Raises:
            ValueError: If the API key cannot be provided or read.
        """
        if wandb_api_key:
            if os.path.isfile(wandb_api_key):
                # Read API key from file
                with open(wandb_api_key, 'r') as f:
                    api_key = f.read().strip()
                wandb.login(key=api_key)
            else:
                # Treat it as a direct API key string
                wandb.login(key=wandb_api_key)
        else:
            # Try to login with environment variable or pre-existing config
            try:
                wandb.login()
            except wandb.errors.UsageError as e:
                raise ValueError("W&B API key not configured. Please provide it as an argument, "
                                 "specify a file, or set it as an environment variable.") from e