"""合成データで、帯の抽出・節点の重複除去・深さごとの最近点選定を確認する。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sheetpile_profile import (  # noqa: E402
    NodeRow,
    depth_key,
    load_folder_series,
    phase_label,
    plot_folder,
    read_displacement_table,
    select_profile,
    series_warnings,
)

import matplotlib.pyplot as plt  # noqa: E402


def _table(lines: list[str]) -> str:
    header = (
        "             構造要素\t   節点\tローカル番号\t"
        "                Y [m]\t                 Z [m]\t               u_x [m]"
    )
    return "\ufeff" + header + "\r\n" + "\r\n".join(lines) + "\r\n"


class SelectProfileTests(unittest.TestCase):
    def test_drops_points_outside_the_band(self) -> None:
        rows = [
            NodeRow("in", 18.0, -2.0, 0.2),
            NodeRow("edge", 18.009, 0.0, 0.3),
            NodeRow("out_high", 18.02, 1.0, 9.0),
            NodeRow("out_low", 17.98, 1.0, 8.0),
        ]
        points = select_profile(rows, y_target=18.0, tolerance=0.01)
        self.assertEqual([point.z for point in points], [-2.0, 0.0])
        self.assertAlmostEqual(points[0].ux, 0.2)
        self.assertAlmostEqual(points[1].ux, 0.3)

    def test_same_node_is_counted_once(self) -> None:
        rows = [
            NodeRow("1", 18.0, 0.0, 1.0),
            NodeRow("1", 18.0, 0.0, 1.0),
            NodeRow("2", 18.0, 0.0, 3.0),
        ]
        points = select_profile(rows, y_target=18.0, tolerance=0.01)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0].ux, 2.0)

    def test_duplicate_node_keeps_the_first_row(self) -> None:
        rows = [
            NodeRow("1", 18.0, -1.0, 1.0),
            NodeRow("1", 18.0, -1.0, 5.0),
        ]
        points = select_profile(rows, y_target=18.0, tolerance=0.01)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0].ux, 1.0)

    def test_same_depth_keeps_closest_y(self) -> None:
        rows = [
            NodeRow("far", 18.008, -1.0, 5.0),
            NodeRow("near", 18.001, -1.0, 1.5),
            NodeRow("other", 17.995, 2.0, 0.4),
            NodeRow("outside", 19.0, -1.0, 100.0),
        ]
        points = select_profile(rows, y_target=18.0, tolerance=0.01)
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0].z, -1.0)
        self.assertAlmostEqual(points[0].ux, 1.5)
        self.assertAlmostEqual(points[1].z, 2.0)
        self.assertAlmostEqual(points[1].ux, 0.4)

    def test_equal_distance_averages_ux(self) -> None:
        rows = [
            NodeRow("a", 18.002, 1.25, 1.0),
            NodeRow("b", 17.998, 1.25, 3.0),
            NodeRow("c", 18.008, 1.25, 100.0),
        ]
        points = select_profile(rows, y_target=18.0, tolerance=0.01)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0].ux, 2.0)
        self.assertAlmostEqual(points[0].z, 1.25)

    def test_z_rounded_to_1e_6_is_one_depth(self) -> None:
        self.assertEqual(depth_key(1.0000004), depth_key(0.9999996))
        rows = [
            NodeRow("a", 18.0, 1.0000004, 1.0),
            NodeRow("b", 18.004, 0.9999996, 9.0),
        ]
        points = select_profile(rows, y_target=18.0, tolerance=0.01)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0].ux, 1.0)
        self.assertAlmostEqual(points[0].z, 1.0000004)


class TableReadTests(unittest.TestCase):
    def test_read_bom_tab_table_then_select(self) -> None:
        text = _table(
            [
                r"Plate\_1\_1	10	1	1.80020000000000E+001	0.00000000000000E+000	1.00000000000000E+000",
                r"Plate\_1\_1	10	5	1.80020000000000E+001	0.00000000000000E+000	1.00000000000000E+000",
                r"(鋼矢板)	11	1	1.79980000000000E+001	0.00000000000000E+000	3.00000000000000E+000",
                r"Plate\_1\_2	12	1	1.90000000000000E+001	1.00000000000000E+000	9.00000000000000E+000",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.TXT"
            path.write_bytes(text.encode("utf-8"))
            rows = read_displacement_table(path)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0].node, "10")
        points = select_profile(rows, y_target=18.0, tolerance=0.01)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0].ux, 2.0)

    def test_phase_labels_follow_filename_code_point(self) -> None:
        names = [
            "Calculation results Plate 載荷 [Phase_3] 3135 全変位の表.TXT",
            "Calculation results Plate 矢板施工 [Phase_2] 2116 全変位の表.TXT",
            "Calculation results Plate 掘削4 [Phase_7] 7337 全変位の表.TXT",
            "Calculation results Plate 掘削1 [Phase_4] 4147 全変位の表.TXT",
            "Calculation results Plate 掘削3 [Phase_6] 6198 全変位の表.TXT",
            "Calculation results Plate 掘削2 [Phase_5] 5164 全変位の表.TXT",
        ]
        labels = [phase_label(name) for name in sorted(names)]
        self.assertEqual(
            labels,
            ["掘削1", "掘削2", "掘削3", "掘削4", "矢板施工", "載荷"],
        )
        self.assertEqual(
            phase_label(
                "Calculation results Plate 矢板施工 [Phase_2] 2116 全変位の表 1.TXT"
            ),
            "矢板施工",
        )

    def test_folder_series_follow_filename_order(self) -> None:
        header = (
            "構造要素\t節点\tローカル番号\tY [m]\tZ [m]\tu_x [m]\n"
        )
        files = {
            "Calculation results Plate 載荷 [Phase_3] 3135 全変位の表.TXT": "1",
            "Calculation results Plate 掘削1 [Phase_4] 4147 全変位の表.TXT": "2",
        }
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, node in files.items():
                (folder / name).write_text(
                    header + f"Plate\t{node}\t1\t18\t0\t0.1\n",
                    encoding="utf-8",
                )
            series = load_folder_series(folder, y_target=18.0, tolerance=0.01)
        self.assertEqual([item.label for item in series], ["掘削1", "載荷"])
        self.assertTrue(all(item.n_points == 1 for item in series))


class PlotTests(unittest.TestCase):
    def test_plot_skips_empty_series_and_keeps_legend_order(self) -> None:
        header = "構造要素\t節点\tローカル番号\tY [m]\tZ [m]\tu_x [m]\n"
        files = {
            "Calculation results Plate 載荷 [Phase_3] 3135 全変位の表.TXT": "18\t1\t0.2",
            "Calculation results Plate 矢板施工 [Phase_2] 2116 全変位の表.TXT": "30\t1\t0.9",
            "Calculation results Plate 掘削2 [Phase_5] 5164 全変位の表.TXT": "18\t0\t0.3",
            "Calculation results Plate 掘削1 [Phase_4] 4147 全変位の表.TXT": "18\t-1\t0.1",
        }
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, values in files.items():
                (folder / name).write_text(
                    header + f"Plate\t1\t1\t{values}\n",
                    encoding="utf-8",
                )
            output = folder / "out.png"
            fig, series = plot_folder(
                folder,
                y_target=18.0,
                tolerance=0.01,
                output_path=output,
            )
            labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
            self.assertEqual(labels, ["掘削1", "掘削2", "載荷"])
            self.assertEqual(fig.axes[0].get_xlabel(), "u_x [m]")
            self.assertEqual(fig.axes[0].get_ylabel(), "Z [m]")
            warnings = series_warnings(series)
            self.assertEqual(len(warnings), 1)
            self.assertIn("矢板施工", warnings[0])
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
