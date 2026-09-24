import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import h5py


class InventoryCounts(unittest.TestCase):
    def test_duplicate_files_multidemo_containers_and_unknown_layouts(self):
        script=Path(__file__).resolve().parents[1]/'scripts/count_trajectories.py'
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            with h5py.File(root/'native.hdf5','w') as f:
                f.create_dataset('actions',data=[[1],[2]])
            shutil.copyfile(root/'native.hdf5',root/'duplicate.hdf5')
            with h5py.File(root/'container.hdf5','w') as f:
                for i in range(2):f.create_dataset(f'data/demo_{i}/actions',data=[[1]])
            with h5py.File(root/'unknown.hdf5','w') as f:
                f.create_dataset('unrelated',data=[1])
            output=root/'inventory.json'
            subprocess.run([sys.executable,str(script),str(root),'--output',str(output)],
                           check=True,capture_output=True,text=True)
            data=json.loads(output.read_text())
            self.assertEqual(data['files'],4)
            self.assertEqual(data['contained_trajectories'],4)
            self.assertEqual(data['byte_unique_trajectories'],3)
            self.assertEqual(sum('exact_duplicate_of' in r for r in data['records']),1)
            self.assertEqual(sum('error' in r for r in data['records']),1)
