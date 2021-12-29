import shutil
import subprocess
import vapoursynth as vs
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Union, Tuple


from ..utils.spathlib import SPath
from ..dataclasses import IndexFileType


core = vs.core


class DVDIndexer(ABC):
    """Abstract DVD indexer interface."""

    def __init__(
        self, bin_path: Union[SPath, str], vps_indexer: Callable[..., vs.VideoNode],
        ext: str, force: bool = True, **indexer_kwargs: Any
    ) -> None:
        self.bin_path = SPath(bin_path)
        self.vps_indexer = vps_indexer
        self.ext = ext
        self.force = force
        self.indexer_kwargs = indexer_kwargs
        super().__init__()

    @abstractmethod
    def get_cmd(self, files: List[SPath], output: SPath) -> List[str]:
        """Returns the indexer command"""
        raise NotImplementedError

    @abstractmethod
    def get_info(self, index_path: SPath, file_idx: int = 0) -> IndexFileType:
        """Returns info about the indexing file"""
        raise NotImplementedError

    @abstractmethod
    def update_video_filenames(self, index_path: SPath, filepaths: List[SPath]) -> None:
        raise NotImplementedError

    def _check_bin_path(self) -> SPath:
        if not shutil.which(str(self.bin_path)):
            raise FileNotFoundError(f'DVDIndexer: `{self.bin_path}` was not found!')
        return self.bin_path

    def index(self, files: List[Path], output: Path, *cmd_args: str) -> None:
        subprocess.run(
            list(map(str, self.get_cmd(files, output))) + list(cmd_args),
            check=True, text=True, encoding='utf-8',
            stdout=subprocess.PIPE, cwd=files[0].parent
        )

    def get_idx_file_path(self, path: Path) -> Path:
        return path.with_suffix(f'.{self.ext}')

    def file_corrupted(self, index_path: SPath) -> None:
        if self.force:
            try:
                index_path.unlink()
            except OSError:
                raise RuntimeError("IsoFile: Index file corrupted, tried to delete it and failed.")
        else:
            raise RuntimeError("IsoFile: Index file corrupted! Delete it and retry.")

    @staticmethod
    def _split_lines(buff: List[str]) -> Tuple[List[str], List[str]]:
        split_idx = buff.index('')
        return buff[:split_idx], buff[split_idx + 1:]
