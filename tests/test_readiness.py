"""Downloaded assets and installed packages must never manufacture M0 READY."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


class Readiness(unittest.TestCase):
    def test_static_success_without_physics_cannot_be_ready(self):
        path=Path(__file__).resolve().parents[1]/'scripts/summarize_m0.py'
        spec=importlib.util.spec_from_file_location('m0_summary',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'outputs/environment_check';out.mkdir(parents=True)
            manifest=root/'assets/manifests';manifest.mkdir(parents=True)
            (manifest/'object_registry.json').write_text(json.dumps({'objects':[{'asset_id':str(i)} for i in range(30)]}))
            for name in ['offline_asset_check.json','storage_audit.json','github_sync.json']:
                (out/name).write_text(json.dumps({'status':'PASS'}))
            (out/'environment_versions.json').write_text(json.dumps({'installation_status':'PASS'}))
            (out/'gpu_check.json').write_text(json.dumps({'rtx_rendering':'PASS'}))
            (out/'cuda_driver_check.json').write_text(json.dumps({'cuInit_returncode':999}))
            module.ROOT=root;module.OUT=out
            with contextlib.redirect_stdout(io.StringIO()):
                code=module.main()
            report=json.loads((out/'m0_summary.json').read_text())
            self.assertEqual(code,2)
            self.assertFalse(report['APFG_M0_READY'])
            self.assertEqual(report['gates']['gpu_compute_and_rendering'],'BLOCKED')
            self.assertEqual(report['physically_verified_ycb_count'],0)


if __name__=='__main__': unittest.main()
