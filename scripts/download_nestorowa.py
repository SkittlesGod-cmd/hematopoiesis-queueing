"""Download Nestorowa et al. 2016 data.

Used for structural cross-check only (single timepoint).
"""

from __future__ import annotations

import os
import shutil
import ssl
import urllib.request
from pathlib import Path

# Create unverified SSL context for environments with certificate issues
_SSL_CONTEXT = ssl._create_unverified_context()

# Nestorowa data hosted on blood.stemcells.cam.ac.uk
BASE_URL = "https://blood.stemcells.cam.ac.uk/data/"
FILES = {
    "nestorowa_corrected_log2_transformed_counts.txt.gz": "nestorowa_corrected_log2_transformed_counts.txt.gz",
}


def download_nestorowa(output_dir: str | Path | None = None) -> Path:
    """Download Nestorowa data files.

    Parameters
    ----------
    output_dir : str or Path, optional
        Destination directory. Defaults to scripts/data/raw/nestorowa/.

    Returns
    -------
    Path
        Directory containing downloaded files.
    """
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "data" / "raw" / "nestorowa"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, target_name in FILES.items():
        target_path = output_dir / target_name
        if target_path.exists():
            print(f"  Already exists: {target_name}")
            continue

        url = BASE_URL + filename
        print(f"  Downloading {filename}...")
        try:
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=_SSL_CONTEXT)
            )
            with opener.open(url) as response, open(target_path, "wb") as out:
                shutil.copyfileobj(response, out)
            print(f"  -> {target_path}")
        except Exception as e:
            print(f"  ERROR downloading {filename}: {e}")
            print(f"  Nestorowa data may need manual download from the publication.")
            raise

    print(f"\nNestorowa data ready in: {output_dir}")
    return output_dir


if __name__ == "__main__":
    print("Downloading Nestorowa et al. 2016...")
    download_nestorowa()
