"""Bounded, resumable HTTPS transfers and safe archive extraction."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def download(url, destination, expected_sha256=None):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(path.suffix + '.json')
    if path.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['url'] == url and saved['sha256'] == sha256(path):
            return saved
    # HEAD both checks connectivity and gives the peak-space estimate.
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=40) as response:
        size = int(response.headers.get('Content-Length', '0'))
    if shutil.disk_usage(path.parent).free < size * 5 + 20 * 2**30:
        raise RuntimeError('Insufficient free space including 20 GiB reserve')
    partial = path.with_suffix(path.suffix + '.part')
    if path.exists() and not partial.exists():
        path.rename(partial)
    # curl validates TLS and refuses unsafe range resumption if unsupported.
    subprocess.run(['curl', '--fail', '--location', '--retry', '4', '--retry-delay', '3',
                    '--connect-timeout', '30', '--max-time', '1800', '--continue-at', '-',
                    '--output', str(partial), url], check=True)
    if size and partial.stat().st_size != size:
        raise RuntimeError(f'Content length mismatch: {partial}')
    digest = sha256(partial)
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(f'SHA256 mismatch: {partial}')
    partial.replace(path)
    saved = {'url': url, 'sha256': digest, 'bytes': path.stat().st_size,
             'integrity': 'publisher_sha256' if expected_sha256 else 'local_sha256_and_content_length'}
    receipt.write_text(json.dumps(saved, indent=2) + '\n')
    return saved


def extract_tar(archive, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()  # gzip CRC and tar structure checked during extraction
        for m in members:
            target = (directory / m.name).resolve()
            if not target.is_relative_to(directory) or m.issym() or m.islnk() or m.isdev():
                raise ValueError(f'Unsafe archive member: {m.name}')
        tar.extractall(directory, members=members)
