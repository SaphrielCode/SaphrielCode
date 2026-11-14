"""Binary restoration must be exact and must never overwrite a different artifact."""
import unittest,tempfile,hashlib,importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('artifact_restore',ROOT/'tools/restore_published_artifacts.py');restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)
class ArtifactTests(unittest.TestCase):
 def test_exact_restore_and_existing_protection(self):
  with tempfile.TemporaryDirectory() as folder:
   previous=restore.ROOT;restore.ROOT=Path(folder).resolve()
   try:
    parts=[]
    for i,data in enumerate((b'abc',b'def')):
     name=f'p{i}.bin';(restore.ROOT/name).write_bytes(data);parts.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    entry=dict(file='models/model.pt',bytes=6,sha256=hashlib.sha256(b'abcdef').hexdigest(),parts=parts);self.assertTrue(restore.restore(entry));self.assertEqual((restore.ROOT/entry['file']).read_bytes(),b'abcdef');self.assertFalse(restore.restore(entry));(restore.ROOT/entry['file']).write_bytes(b'keep')
    with self.assertRaises(FileExistsError):restore.restore(entry)
    self.assertEqual((restore.ROOT/entry['file']).read_bytes(),b'keep')
   finally:restore.ROOT=previous
 def test_bad_part_and_path_escape(self):
  with tempfile.TemporaryDirectory() as folder:
   previous=restore.ROOT;restore.ROOT=Path(folder).resolve()
   try:
    (restore.ROOT/'part.bin').write_bytes(b'wrong');entry=dict(file='model.pt',bytes=5,sha256='invalid',parts=[dict(path='part.bin',bytes=5,sha256='invalid')])
    with self.assertRaises(ValueError):restore.restore(entry)
    self.assertFalse((restore.ROOT/'model.pt').exists());self.assertFalse((restore.ROOT/'model.pt.restore.tmp').exists())
    with self.assertRaises(ValueError):restore.inside('../outside')
   finally:restore.ROOT=previous
if __name__=='__main__':unittest.main()
