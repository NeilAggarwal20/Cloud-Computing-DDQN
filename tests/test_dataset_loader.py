import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from dataset_loader import CloudWorkloadDataset


class CloudWorkloadDatasetTests(unittest.TestCase):
    def test_current_dataset_is_valid_and_split_is_disjoint(self):
        data = CloudWorkloadDataset(ROOT / 'data' / 'cloud_workloads.csv')
        self.assertEqual(data.record_count, 600)
        self.assertEqual(len(data.train_rows), 480)
        self.assertEqual(len(data.evaluation_rows), 120)
        self.assertEqual(data.duplicate_workload_records, 0)
        self.assertFalse({r['task_id'] for r in data.train_rows} &
                         {r['task_id'] for r in data.evaluation_rows})
        self.assertFalse(data.summary()['target_used'])
        self.assertFalse(data.summary()['vm_id_used'])

    def test_profile_is_seeded_and_expands_with_replacement(self):
        data = CloudWorkloadDataset(ROOT / 'data' / 'cloud_workloads.csv')
        with tempfile.TemporaryDirectory() as temp:
            first = data.write_profile(Path(temp) / 'first.csv', 150, 1234, 'evaluation')
            second = data.write_profile(Path(temp) / 'second.csv', 150, 1234, 'evaluation')
            self.assertEqual(first.read_text().splitlines(), second.read_text().splitlines())
            self.assertEqual(len(first.read_text().splitlines()), 151)


if __name__ == '__main__':
    unittest.main()
