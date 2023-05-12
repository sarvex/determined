import contextlib
import logging
import os
import tempfile
from typing import Iterator, Optional

from hdfs.client import InsecureClient

from determined.common import util
from determined.common.storage.base import StorageManager, StorageMetadata


class HDFSStorageManager(StorageManager):
    """
    Store and load checkpoints from HDFS.
    """

    def __init__(
        self,
        hdfs_url: str,
        hdfs_path: str,
        user: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ) -> None:
        super().__init__(temp_dir if temp_dir is not None else tempfile.gettempdir())

        self.hdfs_url = hdfs_url
        self.hdfs_path = hdfs_path
        self.user = user

        self.client = InsecureClient(self.hdfs_url, root=self.hdfs_path, user=self.user)

    @util.preserve_random_state
    def post_store_path(self, storage_id: str, storage_dir: str, metadata: StorageMetadata) -> None:
        """post_store_path uploads the checkpoint to hdfs and deletes the original files."""
        try:
            logging.info(f"Uploading storage {storage_id} to HDFS")
            result = self.client.upload(metadata, storage_dir)

            logging.info(f"Uploaded storage {storage_id} to HDFS path {result}")
        finally:
            self._remove_checkpoint_directory(metadata.storage_id)

    @contextlib.contextmanager
    def restore_path(self, metadata: StorageMetadata) -> Iterator[str]:
        logging.info(f"Downloading storage {metadata.storage_id} from HDFS")

        self.download(metadata)

        try:
            yield os.path.join(self._base_path, metadata.storage_id)
        finally:
            self._remove_checkpoint_directory(metadata.storage_id)

    @util.preserve_random_state
    def download(self, metadata: StorageMetadata) -> None:
        self.client.download(metadata.storage_id, self._base_path, overwrite=True)

    @util.preserve_random_state
    def delete(self, metadata: StorageMetadata) -> None:
        logging.info(f"Deleting storage {metadata.storage_id} from HDFS")
        self.client.delete(metadata.storage_id, recursive=True)
