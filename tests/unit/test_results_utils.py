"""
Unit tests for ovo.utils.results_utils

All tests that touch the filesystem use tmp_path (pytest built-in).
Plot tests only verify that each function returns a Figure object and do
not render anything to screen or disk.
"""

import math
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use('Agg')  # headless backend for CI

from matplotlib.figure import Figure

from ovo.utils.results_utils import (
    filter_experiments,
    get_available_dates,
    get_available_methods,
    get_available_noise_levels,
    get_available_scenes,
    load_experiments,
    parse_experiment_name,
    parse_instance_counts,
    parse_statistics_file,
    plot_bar_metrics,
    plot_class_heatmap,
    plot_confusion_matrix,
    plot_instance_counts,
    plot_line_metrics,
    save_figure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stats_file(path: Path, rows: list[str]) -> Path:
    """Write a statistics.txt with the given data rows (header is added)."""
    path.write_text("label, acc, iou, \n" + "".join(rows))
    return path


def _make_instance_file(path: Path, lines: int) -> Path:
    """Write a fake instance_pred file with `lines` non-empty lines."""
    path.write_text("\n".join(str(i) for i in range(lines)) + "\n")
    return path


def _minimal_exp_df(**overrides) -> pd.DataFrame:
    """Return a single-row DataFrame suitable for plot tests."""
    row = {
        "Date": "20260101",
        "SLAM_Config": "GTNoise-T0p01-R0p01",
        "Fusion": "CLIP",
        "Label": "Test",
        "Method": "CLIP_Test",
        "Trans_Noise": 0.01,
        "Rot_Noise": 0.01,
        "mIoU": 0.25,
        "mAcc": 0.35,
        "Num_Instances": 100,
        "Experiment_ID": "20260101_GTNoise-T0p01-R0p01_CLIP_Test",
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# parse_experiment_name
# ---------------------------------------------------------------------------

class TestParseExperimentName:

    def test_standard_4_parts(self):
        result = parse_experiment_name("20251218_GTNoise-T0p001-R0p01_CLIP_Original")
        assert result is not None
        assert result["Date"] == "20251218"
        assert result["SLAM_Config"] == "GTNoise-T0p001-R0p01"
        assert result["Fusion"] == "CLIP"
        assert result["Label"] == "Original"
        assert result["Method"] == "CLIP_Original"

    def test_label_with_underscores(self):
        """Labels spanning multiple underscore-separated parts must be joined."""
        result = parse_experiment_name(
            "20260218_GTNoise-T0p001-R0p01_PE_PE-Spatial-Fusion_Rerun"
        )
        assert result is not None
        assert result["Fusion"] == "PE"
        assert result["Label"] == "PE-Spatial-Fusion_Rerun"
        assert result["Method"] == "PE_PE-Spatial-Fusion_Rerun"

    def test_noise_extraction_small(self):
        result = parse_experiment_name("20260101_GTNoise-T0p001-R0p01_CLIP_Test")
        assert result is not None
        assert math.isclose(result["Trans_Noise"], 0.001)
        assert math.isclose(result["Rot_Noise"], 0.01)

    def test_noise_extraction_large(self):
        result = parse_experiment_name("20260101_GTNoise-T0p05-R0p01_PE_Test")
        assert result is not None
        assert math.isclose(result["Trans_Noise"], 0.05)

    def test_sam3_fusion(self):
        result = parse_experiment_name("20260121_GTNoise-T0p05-R0p01_SAM3_SAM3-Fusion")
        assert result is not None
        assert result["Fusion"] == "SAM3"
        assert result["Label"] == "SAM3-Fusion"

    def test_invalid_too_few_parts(self):
        assert parse_experiment_name("global_correction_test") is None
        assert parse_experiment_name("only_three_parts") is None

    def test_invalid_no_date(self):
        assert parse_experiment_name("notadate_GTNoise-T0p01-R0p01_CLIP_X") is None

    def test_zero_noise_config_is_valid(self):
        """SLAM configs without T/R markers (GT, ORBSLAM3) are valid with zero noise."""
        result = parse_experiment_name("20260101_ORBSLAM3_CLIP_Test")
        assert result is not None
        assert math.isclose(result["Trans_Noise"], 0.0)
        assert math.isclose(result["Rot_Noise"], 0.0)

        result_gt = parse_experiment_name("20260101_GT_CLIP_Test")
        assert result_gt is not None
        assert math.isclose(result_gt["Trans_Noise"], 0.0)


# ---------------------------------------------------------------------------
# parse_statistics_file
# ---------------------------------------------------------------------------

class TestParseStatisticsFile:

    def test_basic_parsing(self, tmp_path):
        f = _make_stats_file(
            tmp_path / "statistics.txt",
            [
                "wall, 0.6, 0.5, \n",
                "chair, 0.9, 0.8, \n",
                "floor, 0.7, 0.6, \n",
            ],
        )
        result = parse_statistics_file(f)
        assert result is not None
        assert math.isclose(result["mIoU"], (0.5 + 0.8 + 0.6) / 3)
        assert math.isclose(result["mAcc"], (0.6 + 0.9 + 0.7) / 3)
        assert len(result["per_class"]) == 3

    def test_nan_rows_excluded_from_mean(self, tmp_path):
        """NaN entries should not count toward mIoU / mAcc."""
        f = _make_stats_file(
            tmp_path / "statistics.txt",
            [
                "wall, 0.5, 0.4, \n",
                "bench, nan, nan, \n",
            ],
        )
        result = parse_statistics_file(f)
        assert result is not None
        assert math.isclose(result["mIoU"], 0.4)
        assert math.isclose(result["mAcc"], 0.5)
        assert len(result["per_class"]) == 2   # nan row still present in per_class

    def test_missing_file_returns_none(self, tmp_path):
        assert parse_statistics_file(tmp_path / "nonexistent.txt") is None

    def test_empty_file_returns_none(self, tmp_path):
        f = tmp_path / "statistics.txt"
        f.write_text("label, acc, iou, \n")   # header only
        assert parse_statistics_file(f) is None

    def test_per_class_dataframe_columns(self, tmp_path):
        f = _make_stats_file(
            tmp_path / "statistics.txt",
            ["wall, 0.6, 0.5, \n"],
        )
        result = parse_statistics_file(f)
        assert {"Class", "Acc", "IoU"}.issubset(result["per_class"].columns)


# ---------------------------------------------------------------------------
# parse_instance_counts
# ---------------------------------------------------------------------------

class TestParseInstanceCounts:

    def test_counts_lines(self, tmp_path):
        exp = tmp_path / "exp1"
        pred = exp / "instance_pred"
        pred.mkdir(parents=True)
        _make_instance_file(pred / "office0.txt", 42)
        assert parse_instance_counts(exp, "office0") == 42

    def test_fallback_to_any_txt(self, tmp_path):
        exp = tmp_path / "exp2"
        pred = exp / "instance_pred"
        pred.mkdir(parents=True)
        _make_instance_file(pred / "room1.txt", 10)
        # Request non-existent scene, should fall back to room1.txt
        assert parse_instance_counts(exp, "office0") == 10

    def test_missing_dir_returns_zero(self, tmp_path):
        assert parse_instance_counts(tmp_path / "no_such_exp", "office0") == 0

    def test_empty_dir_returns_zero(self, tmp_path):
        exp = tmp_path / "exp3"
        (exp / "instance_pred").mkdir(parents=True)
        assert parse_instance_counts(exp, "office0") == 0

    def test_ignores_blank_lines(self, tmp_path):
        exp = tmp_path / "exp4"
        pred = exp / "instance_pred"
        pred.mkdir(parents=True)
        (pred / "office0.txt").write_text("1\n2\n\n3\n\n")
        assert parse_instance_counts(exp, "office0") == 3


# ---------------------------------------------------------------------------
# load_experiments
# ---------------------------------------------------------------------------

class TestLoadExperiments:

    def _make_experiment(self, root: Path, name: str, stats_rows, n_instances: int,
                         dataset: str = "Replica"):
        """Create a fake experiment under root/{dataset}/{name}/ as the loader expects."""
        exp = root / dataset / name
        replica = exp / "replica"
        replica.mkdir(parents=True)
        pred = exp / "instance_pred"
        pred.mkdir(parents=True)
        _make_stats_file(replica / "statistics.txt", stats_rows)
        _make_instance_file(pred / "office0.txt", n_instances)

    def test_loads_valid_experiments(self, tmp_path):
        self._make_experiment(
            tmp_path,
            "20260101_GTNoise-T0p01-R0p01_CLIP_Test",
            ["wall, 0.5, 0.4, \n", "chair, 0.8, 0.7, \n"],
            50,
        )
        df_exp, df_class = load_experiments(tmp_path)
        assert len(df_exp) == 1
        assert df_exp.iloc[0]["Num_Instances"] == 50
        assert math.isclose(df_exp.iloc[0]["mIoU"], (0.4 + 0.7) / 2)
        assert len(df_class) == 2   # one row per class

    def test_skips_non_conforming_folders(self, tmp_path):
        (tmp_path / "global_correction_test").mkdir()
        df_exp, _ = load_experiments(tmp_path)
        assert len(df_exp) == 0

    def test_date_filter(self, tmp_path):
        for date in ["20260101", "20260202"]:
            self._make_experiment(
                tmp_path,
                f"{date}_GTNoise-T0p01-R0p01_CLIP_Test",
                ["wall, 0.5, 0.4, \n"],
                10,
            )
        df_exp, _ = load_experiments(tmp_path, date_filter="20260101")
        assert len(df_exp) == 1
        assert df_exp.iloc[0]["Date"] == "20260101"

    def test_method_filter(self, tmp_path):
        for method_label in ["CLIP_TestA", "PE_TestB"]:
            fusion, label = method_label.split("_", 1)
            self._make_experiment(
                tmp_path,
                f"20260101_GTNoise-T0p01-R0p01_{fusion}_{label}",
                ["wall, 0.5, 0.4, \n"],
                10,
            )
        df_exp, _ = load_experiments(tmp_path, method_filter=["CLIP_TestA"])
        assert len(df_exp) == 1
        assert df_exp.iloc[0]["Method"] == "CLIP_TestA"

    def test_empty_dir_returns_empty_dataframes(self, tmp_path):
        df_exp, df_class = load_experiments(tmp_path)
        assert df_exp.empty
        assert df_class.empty
        # Must still have the expected columns
        assert "mIoU" in df_exp.columns
        assert "Class" in df_class.columns

    def test_required_columns_present(self, tmp_path):
        self._make_experiment(
            tmp_path,
            "20260101_GTNoise-T0p01-R0p01_CLIP_Test",
            ["wall, 0.5, 0.4, \n"],
            5,
        )
        df_exp, df_class = load_experiments(tmp_path)
        exp_cols = {"Date", "SLAM_Config", "Fusion", "Label", "Method",
                    "Trans_Noise", "Rot_Noise", "mIoU", "mAcc", "Num_Instances", "Experiment_ID"}
        assert exp_cols.issubset(df_exp.columns)
        assert {"Class", "IoU", "Acc"}.issubset(df_class.columns)


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

class TestFilterHelpers:

    def _make_df(self) -> pd.DataFrame:
        rows = [
            {"Date": "20260101", "Scene": "office0", "Method": "CLIP_Test",
             "Trans_Noise": 0.01, "Rot_Noise": 0.01},
            {"Date": "20260101", "Scene": "office0", "Method": "PE_Test",
             "Trans_Noise": 0.05, "Rot_Noise": 0.01},
            {"Date": "20260202", "Scene": "room0",   "Method": "CLIP_Test",
             "Trans_Noise": 0.01, "Rot_Noise": 0.01},
        ]
        return pd.DataFrame(rows)

    def test_get_available_dates(self):
        df = self._make_df()
        assert get_available_dates(df) == ["20260101", "20260202"]

    def test_get_available_scenes(self):
        df = self._make_df()
        assert get_available_scenes(df) == ["office0", "room0"]

    def test_get_available_methods(self):
        df = self._make_df()
        assert get_available_methods(df) == ["CLIP_Test", "PE_Test"]

    def test_get_available_noise_levels(self):
        df = self._make_df()
        assert get_available_noise_levels(df) == [0.01, 0.05]

    def test_filter_by_date(self):
        df = self._make_df()
        result = filter_experiments(df, dates=["20260101"])
        assert len(result) == 2
        assert all(result["Date"] == "20260101")

    def test_filter_by_method(self):
        df = self._make_df()
        result = filter_experiments(df, methods=["PE_Test"])
        assert len(result) == 1
        assert result.iloc[0]["Method"] == "PE_Test"

    def test_filter_by_noise(self):
        df = self._make_df()
        result = filter_experiments(df, noise_levels=[0.05])
        assert len(result) == 1
        assert math.isclose(result.iloc[0]["Trans_Noise"], 0.05)

    def test_filter_and_logic(self):
        """Multiple filters are combined with AND."""
        df = self._make_df()
        result = filter_experiments(df, dates=["20260101"], methods=["PE_Test"])
        assert len(result) == 1

    def test_filter_none_means_no_filter(self):
        df = self._make_df()
        result = filter_experiments(df)
        assert len(result) == len(df)

    def test_filter_returns_copy(self):
        """Mutation of filtered result must not affect original."""
        df = self._make_df()
        filtered = filter_experiments(df, dates=["20260101"])
        filtered.loc[filtered.index[0], "Date"] = "MUTATED"
        assert "MUTATED" not in df["Date"].values


# ---------------------------------------------------------------------------
# Plot functions — only check return type
# ---------------------------------------------------------------------------

class TestPlots:

    def _exp_df(self, n=3) -> pd.DataFrame:
        rows = [_minimal_exp_df(
            Trans_Noise=v, Method=m,
            Experiment_ID=f"20260101_office0_GTNoise-T{str(v).replace('.','p')}-R0p01_{m}"
        ).iloc[0] for v, m in [(0.01, "CLIP_A"), (0.05, "CLIP_A"), (0.01, "PE_B")]]
        return pd.DataFrame(rows[:n])

    def _class_df(self) -> pd.DataFrame:
        base = self._exp_df()
        rows = []
        for _, row in base.iterrows():
            for cls in ["wall", "chair"]:
                r = row.to_dict()
                r.update({"Class": cls, "IoU": 0.5, "Acc": 0.6})
                rows.append(r)
        return pd.DataFrame(rows)

    def test_plot_bar_metrics_returns_figure(self):
        fig = plot_bar_metrics(self._exp_df())
        assert isinstance(fig, Figure)

    def test_plot_line_metrics_returns_figure(self):
        fig = plot_line_metrics(self._exp_df())
        assert isinstance(fig, Figure)

    def test_plot_line_metrics_single_metric(self):
        fig = plot_line_metrics(self._exp_df(), metrics=["mIoU"])
        assert isinstance(fig, Figure)

    def test_plot_class_heatmap_returns_figure(self):
        fig = plot_class_heatmap(self._class_df())
        assert isinstance(fig, Figure)

    def test_plot_class_heatmap_acc(self):
        fig = plot_class_heatmap(self._class_df(), value="Acc")
        assert isinstance(fig, Figure)

    def test_plot_instance_counts_returns_figure(self):
        fig = plot_instance_counts(self._exp_df())
        assert isinstance(fig, Figure)

    def test_plot_confusion_matrix_missing_returns_none(self, tmp_path):
        result = plot_confusion_matrix("nonexistent_experiment", tmp_path)
        assert result is None

    def test_plot_confusion_matrix_with_image(self, tmp_path):
        import matplotlib.pyplot as plt
        import numpy as np

        exp_id = "20260101_office0_GTNoise-T0p01-R0p01_CLIP_Test"
        replica = tmp_path / exp_id / "replica"
        replica.mkdir(parents=True)
        # Save a tiny PNG so the function can load it
        fig_dummy, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(np.zeros((4, 4, 3)))
        fig_dummy.savefig(replica / "confmat.png")
        plt.close(fig_dummy)

        fig = plot_confusion_matrix(exp_id, tmp_path)
        assert isinstance(fig, Figure)

    def test_save_figure(self, tmp_path):
        fig = plot_bar_metrics(_minimal_exp_df())
        out = tmp_path / "test_out.png"
        save_figure(fig, out, dpi=72)
        assert out.exists()
        assert out.stat().st_size > 0
