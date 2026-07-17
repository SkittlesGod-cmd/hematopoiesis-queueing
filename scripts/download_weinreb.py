import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path("data/raw/weinreb")

FILES = [
    (
        "normed_counts",
        "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_normed_counts.mtx.gz",
    ),
    (
        "gene_names",
        "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_gene_names.txt.gz",
    ),
    (
        "clone_matrix",
        "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_clone_matrix.mtx.gz",
    ),
    (
        "metadata",
        "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_metadata.txt.gz",
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
    print(f"Weinreb download complete  ({len(results)} files, {failures} failures)")
    print()
    for name, filename, status in results:
        print(f"  {name:20s}  {filename:45s}  {status}")


if __name__ == "__main__":
    main()
