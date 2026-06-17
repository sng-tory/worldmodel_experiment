from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ldwma.datasets.lerobot_so100 import LeRobotSO100Dataset


class SO100DataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        dataset_paths: list[str],
        batch_size: int = 8,
        target_height: int = 320,
        target_width: int = 512,
        traj_len: int = 16,
        pad: bool = True,
        camera_key: str = "auto",
        val_fraction: float = 0.05,
        seed: int = 0,
        downsample: int = 1,
        use_language: bool = False,
        fps: int | None = None,
        frame_stride: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.dataset_paths = dataset_paths
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.traj_len = traj_len
        self.pad = pad
        self.camera_key = camera_key
        self.val_fraction = val_fraction
        self.seed = seed
        self.downsample = downsample
        self.use_language = use_language
        self.fps = fps
        self.frame_stride = frame_stride
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset = LeRobotSO100Dataset(
            root=self.root,
            dataset_paths=self.dataset_paths,
            train=True,
            traj_len=self.traj_len,
            target_height=self.target_height,
            target_width=self.target_width,
            pad=self.pad,
            camera_key=self.camera_key,
            val_fraction=self.val_fraction,
            seed=self.seed,
            downsample=self.downsample,
            use_language=self.use_language,
            fps=self.fps,
            frame_stride=self.frame_stride,
        )
        self.val_dataset = LeRobotSO100Dataset(
            root=self.root,
            dataset_paths=self.dataset_paths,
            train=False,
            traj_len=self.traj_len,
            target_height=self.target_height,
            target_width=self.target_width,
            pad=self.pad,
            camera_key=self.camera_key,
            val_fraction=self.val_fraction,
            seed=self.seed,
            downsample=self.downsample,
            use_language=self.use_language,
            fps=self.fps,
            frame_stride=self.frame_stride,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
