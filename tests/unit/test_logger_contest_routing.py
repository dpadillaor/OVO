import unittest
import os
import sys
import tempfile

WORKTREE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, WORKTREE_ROOT)

from ovo.entities.logger import Logger, _is_contest_stat, _CONTEST_KF_STATS


class TestContestStatRouting(unittest.TestCase):
    """Las señales del contest (timing t_contest_* + telemetría Tier 2) van a logger/contest/."""

    def test_is_contest_stat(self):
        self.assertTrue(_is_contest_stat("t_contest_report"))
        self.assertTrue(_is_contest_stat("n_matched"))
        self.assertTrue(_is_contest_stat("n_robos"))
        self.assertFalse(_is_contest_stat("t_sam"))
        self.assertFalse(_is_contest_stat("n_matches"))   # distinto de n_matched (Tier2)

    def test_kf_stats_set_complete(self):
        expected = {"n_matched", "n_pre_assign", "n_used", "n_orphans", "n_births", "n_robos"}
        self.assertEqual(_CONTEST_KF_STATS, expected)

    def test_write_stats_routes_to_contest_subdir(self):
        with tempfile.TemporaryDirectory() as d:
            lg = Logger(d)
            # una señal contest y una normal
            lg.log_ovo_stats({"n_matched": 42, "n_robos": 3, "t_sam": 0.1})
            lg.stats["t_contest_report"] = [1.5]
            lg.write_stats()
            base = lg.output_path
            # contest -> logger/contest/
            self.assertTrue((base / "logger" / "contest" / "n_matched.log").exists())
            self.assertTrue((base / "logger" / "contest" / "n_robos.log").exists())
            self.assertTrue((base / "logger" / "contest" / "t_contest_report.log").exists())
            # normal -> logger/
            self.assertTrue((base / "logger" / "t_sam.log").exists())
            self.assertFalse((base / "logger" / "n_matched.log").exists())
            # contenido correcto
            self.assertEqual((base / "logger" / "contest" / "n_matched.log").read_text(), "42")

    def test_dirs_created(self):
        with tempfile.TemporaryDirectory() as d:
            lg = Logger(d)
            self.assertTrue((lg.output_path / "logger" / "contest").is_dir())
            self.assertTrue((lg.output_path / "fusion" / "contest").is_dir())


if __name__ == "__main__":
    unittest.main()
