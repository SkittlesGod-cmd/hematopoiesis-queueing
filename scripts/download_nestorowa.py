import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path("data/raw/nestorowa")

FILES = [
    (
        "expression_matrix",
        "http://blood.stemcells.cam.ac.uk/data/coordinates_gene_counts_flow_cytometry.txt.gz",
    ),
    (
        "cell_type_annotations",
        "http://blood.stemcells.cam.ac.uk/data/all_cell_types.txt",
    ),
    (
        "log2_counts_wolf",
        "http://blood.stemcells.cam.ac.uk/data/nestorowa_corrected_log2_transformed_counts.txt",
    ),
    (
        "population_annotation_wolf",
        "http://blood.stemcells.cam.ac.uk/data/nestorowa_corrected_population_annotation.txt",
    ),
]


def download_file(name: str, url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return False

    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to download {name} from {url}: {e}")

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))

    if dest.stat().st_size == 0:
        raise IOError(f"Downloaded {name} but file is empty: {dest}")

    return True


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    failures = 0

    for name, url in FILES:
        dest = DATA_DIR / os.path.basename(url)
        try:
            downloaded = download_file(name, url, dest)
            status = "downloaded" if downloaded else "already exists"
            results.append((name, dest.name, status))
        except Exception as e:
            results.append((name, dest.name, f"FAILED: {e}"))
            failures += 1

    print()
    print(f"Nestorowa download complete  ({len(results)} files, {failures} failures)")
    print()
    for name, filename, status in results:
        print(f"  {name:25s}  {filename:45s}  {status}")


if __name__ == "__main__":
    main()
