"""Security and data-integrity checks that do not need an operational GPU."""
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from apfg.download import extract_tar


class ArchiveSafety(unittest.TestCase):
    def test_archive_cannot_escape_project_destination(self):
        with tempfile.TemporaryDirectory() as d:
            archive=Path(d)/'bad.tgz'
            with tarfile.open(archive,'w:gz') as tar:
                info=tarfile.TarInfo('../escaped.obj');info.size=3
                tar.addfile(info,io.BytesIO(b'bad'))
            with self.assertRaises(ValueError):
                extract_tar(archive,Path(d)/'safe')
            self.assertFalse((Path(d)/'escaped.obj').exists())

    def test_symlink_archive_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            archive=Path(d)/'link.tgz'
            with tarfile.open(archive,'w:gz') as tar:
                info=tarfile.TarInfo('escape');info.type=tarfile.SYMTYPE;info.linkname='/etc'
                tar.addfile(info)
            with self.assertRaises(ValueError):
                extract_tar(archive,Path(d)/'safe')


if __name__=='__main__':
    unittest.main()
