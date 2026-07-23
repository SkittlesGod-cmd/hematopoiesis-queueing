"""Download Weinreb et al. 2020 data from GEO (GSE140802)."""

from __future__ import annotations

import shutil
import ssl
import urllib.request
from pathlib import Path

# Create unverified SSL context for environments with certificate issues
_SSL_CONTEXT = ssl._create_unverified_context()

BASE_URL = "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/"
FILES = [
    "stateFate_inVitro_normed_counts.mtx.gz",
    "stateFate_inVitro_gene_names.txt.gz",
    "stateFate_inVitro_metadata.txt.gz",
    "stateFate_inVitro_clone_matrix.mtx.gz",
]
# Target file names (same as source)
TARGET_NAMES = FILES


def download_weinreb(output_dir: str | Path | None = None) -> Path:
    """Download Weinreb data files to output directory.

    Parameters
    ----------
    output_dir : str or Path, optional
        Destination directory. Defaults to scripts/data/raw/weinreb/
        relative to this script's location.

    Returns
    -------
    Path
        Directory containing downloaded files.
    """
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "data" / "raw" / "weinreb"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_name, target_name in zip(FILES, TARGET_NAMES):
        target_path = output_dir / target_name
        if target_path.exists():
            print(f"  Already exists: {target_name}")
            continue

        url = BASE_URL + source_name
        print(f"  Downloading {source_name}...")
        try:
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=_SSL_CONTEXT)
            )
            with opener.open(url) as response, open(target_path, "wb") as out:
                shutil.copyfileobj(response, out)
            print(f"  -> {target_path}")
        except Exception as e:
            print(f"  ERROR downloading {source_name}: {e}")
            raise

    print(f"\nWeinreb data ready in: {output_dir}")
    return output_dir


if __name__ == "__main__":
    print("Downloading Weinreb et al. 2020 (GSE140802)...")
    download_weinreb()
