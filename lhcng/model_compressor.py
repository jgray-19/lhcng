from pathlib import Path

import tfs
from omc3.model.constants import (
    TWISS_AC_DAT,
    TWISS_ADT_DAT,
    TWISS_DAT,
    TWISS_ELEMENTS_DAT,
)

class ModelCompressor:
    """Context manager for compressing and decompressing model TFS files."""
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        # Build file paths
        dat_path = self.model_dir / TWISS_DAT
        acd_path = self.model_dir / TWISS_AC_DAT
        adt_path = self.model_dir / TWISS_ADT_DAT
        elements_path = self.model_dir / TWISS_ELEMENTS_DAT

        self.model_files = [dat_path, elements_path]
        if acd_path.exists():
            self.model_files.append(acd_path)
        if adt_path.exists():
            self.model_files.append(adt_path)

    def compress_file(self, uncompressed_path: Path) -> Path:
        """Compress a TFS file and remove the uncompressed file."""
        compressed_path = uncompressed_path.with_suffix(".tfs.bz2")
        df = tfs.read(uncompressed_path)
        tfs.write(compressed_path, df)
        uncompressed_path.unlink()
        return compressed_path

    def decompress_file_to(self, uncompressed_path: Path) -> Path:
        """Decompress a TFS file and remove the compressed file."""
        compressed_path = uncompressed_path.with_suffix(".tfs.bz2")
        df = tfs.read(compressed_path)
        tfs.write(uncompressed_path, df)
        compressed_path.unlink()
        return uncompressed_path

    def compress_model(self) -> None:
        """Compress all model files."""
        for model_file in self.model_files:
            self.compress_file(model_file)

    def decompress_model(self) -> None:
        """Decompress all model files."""
        for model_file in self.model_files:
            self.decompress_file_to(model_file)

    @staticmethod
    def compress_model_folder(model_dir) -> None:
        """Compress all TFS files in the model folder without decompressing."""
        mc = ModelCompressor(model_dir)
        mc.compress_model()

    def __enter__(self):
        self.decompress_model()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.compress_model()
        return False
