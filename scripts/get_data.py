"""Download COLMAP's south-building dataset and extract the 30 frames used here.

    python scripts/get_data.py        # -> data/P1180141.JPG ... data/P1180170.JPG

Each file is checked against the SHA-256 below, so results are reproducible
from exactly the same inputs.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "P1180141.JPG": "be3e0dff291078c936462a00c5fd9c28c96607654395fbd320769ae0b3732b3f",
    "P1180142.JPG": "ac400c467eb5240d4bceea84f7291fd1798079aca75cfd76eafcce5656fa6346",
    "P1180143.JPG": "92d887729eaf159bcc3900e454e089fdfd810b6ca527eefbd9af040decb776d4",
    "P1180144.JPG": "9b160183250da6bac0c025a3635508c544c9aaa85cecc56da20dbbeef0eb9f63",
    "P1180145.JPG": "8dbe80258e34e7d676cc25c4ae6f14374640456b5197d289663e075a80dd785b",
    "P1180146.JPG": "9589a333091bacc27f71499ef0efad3c3e2a8d5eb28067d19367f3c5e368ad3e",
    "P1180147.JPG": "afb5b97398083c17434264ecae7641fad27b0e2cb8f173d65b58112de81545bb",
    "P1180148.JPG": "f3912b040d30bbf1e70723c3644df8d712961d5fed9f2c6fa8e57c9bf5660cb6",
    "P1180149.JPG": "5714aad939d47ff64b3c9d017811ba4ea81a280880807b6c106301875fb35daf",
    "P1180150.JPG": "d8d996dfc0bbcfd602ae9cc948eaedbaf0f170698963dd23e0a374680eca1ff0",
    "P1180151.JPG": "6911d703ad3786b792053321c88d63974498eebad0a6a3ea828ba8f46c4d92fd",
    "P1180152.JPG": "dfafd450716c7bfc8076602c6c31338bf6327eb77348d217345a17694d7dfede",
    "P1180153.JPG": "5350d618b91c516423f975f32d74115396e5ef9422fcea928bb6cc6d22364b30",
    "P1180154.JPG": "70d06a0820eac49ee5b1976532aa4aa2d003554bece613ea7355ead5c88cfacc",
    "P1180155.JPG": "457166a16739cf2b15f1985344298684df475434a8653f61ccc4e0ec18f50b1e",
    "P1180156.JPG": "7eb19f9cfc3b1ca342f2f2c8999c89f5bee47c6542d5f8b0fb3e6b2fc5d668a5",
    "P1180157.JPG": "f14c1e1915b79312b2328475dc91ec39066f3c184ec85a6ae81e0c19a5ab481e",
    "P1180158.JPG": "d9159f4194ff75c6afe8d8b05fdefffa238f7657df1a70b63cf5aec4fc9572c9",
    "P1180159.JPG": "4eed5ee2459747f86d9a15a758355c6637d896e70e48ac6f9aa510b72a77fe70",
    "P1180160.JPG": "ea0bc7b0f4bbe41cd2237c0d7e7e647326ce1322dee0975ccb5f76dc074bef41",
    "P1180161.JPG": "3dec9fe3989fa390df542da2c3d912a3119c58cdb725e2ae04fe83bd9a6a82a1",
    "P1180162.JPG": "d9a6fe7d29ca97079bc54a582ab32f15cbfe59155c2363ae1fd2f4d5a57754dc",
    "P1180163.JPG": "f014473bd8133f940a338d1d1343c4866cbf0d1eb77c4315dd4fd6ca120131d4",
    "P1180164.JPG": "683dc191ea183686a4f3d4dd25a520ea70cb689cb7cf5671365652d0302e4268",
    "P1180165.JPG": "c71d6f0e688e0ab33640ed75fb42e182a8d1cb3aef548015aa03c754c6c2f89d",
    "P1180166.JPG": "5b9578c2d05d8b0c69176b6123b303618061a6377639675534e86912042caa07",
    "P1180167.JPG": "16362bf0c74f68bdf4c23a1f836062903786050fd3cc7d4ff7ae74c2f84fd2bb",
    "P1180168.JPG": "c9358614c07b9a7347ea9023182f300592831dbe540ace03b591be807bcdb09d",
    "P1180169.JPG": "b71ca072500090cbf70ac3ad9d5acb1a8393e25bb847f21f4daac81610313304",
    "P1180170.JPG": "80a9d4b872fadd3c7a4e58058f4cbdad524dd2f976079bdbb60391d56b33f4ce",}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    missing = [n for n, d in FILES.items() if not (data / n).exists() or sha256(data / n) != d]
    if not missing:
        print(f"data/ already has all {len(FILES)} verified frames")
        return 0
    zpath = ROOT / "work" / "south-building.zip"
    zpath.parent.mkdir(exist_ok=True)
    if not zpath.exists():
        print(f"downloading {URL} (~420 MB)")
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        members = {Path(m).name: m for m in z.namelist() if Path(m).name in FILES}
        for name in missing:
            (data / name).write_bytes(z.read(members[name]))
    bad = [n for n in FILES if sha256(data / n) != FILES[n]]
    if bad:
        print(f"checksum mismatch: {bad}")
        return 1
    print(f"extracted and verified {len(FILES)} frames into data/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
