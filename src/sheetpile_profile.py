"""PLAXIS の全変位表から、指定した Y の水平変位プロファイルを作る。

全変位表は UTF-8 BOM のタブ区切りで、列は
``構造要素 / 節点 / ローカル番号 / Y [m] / Z [m] / u_x [m]`` です。
同じ節点がローカル番号違いで繰り返されるので、先に節点を 1 行へ落とします。
そのあと ``|Y - Y_target| <= tolerance`` の帯に絞り、Z を 1e-6 m で丸めた
深さごとに、目標 Y へ最も近い点だけを残します。距離が同じ点は ``u_x`` を平均します。
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Z を 1e-6 m で丸めて、同じ深さの組にする。
Z_ROUND_DIGITS = 6
# 目標 Y までの距離を同じとみなす幅。座標の丸め誤差より広く、別の節点の差より狭い。
DISTANCE_TIE_M = 1e-9

_PHASE_RE = re.compile(r"Plate\s+(.+?)\s+\[")
_REQUIRED_COLUMNS = ("節点", "Y [m]", "Z [m]", "u_x [m]")

# 段階名で色を固定し、陸側と川表で同じ色にする。
PHASE_COLORS: dict[str, str] = {
    "掘削1": "#0072B2",
    "掘削2": "#E69F00",
    "掘削3": "#009E73",
    "掘削4": "#D55E00",
    "矢板施工": "#CC79A7",
    "載荷": "#56B4E9",
}

_HIRAGINO_FAMILIES = (
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Hiragino Kaku Gothic Pro",
    "Hiragino Maru Gothic ProN",
    "Hiragino Maru Gothic Pro",
)
_NOTO_CJK_FAMILIES = (
    "Noto Sans CJK JP",
    "Noto Sans CJK",
    "Noto Sans JP",
)
_NOTO_CJK_FILES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
)

_configured_font: str | None = None


@dataclass(frozen=True)
class NodeRow:
    """全変位表の 1 行。節点の重複はまだ残している。"""

    node: str
    y: float
    z: float
    ux: float


@dataclass(frozen=True)
class ProfilePoint:
    """プロファイルに残した 1 点。``ux`` の単位は m。"""

    z: float
    ux: float


@dataclass(frozen=True)
class SeriesResult:
    """ファイル 1 つ分の系列。採用点が 0 のときも残す。"""

    label: str
    path: Path
    points: tuple[ProfilePoint, ...]

    @property
    def n_points(self) -> int:
        return len(self.points)


def depth_key(z: float) -> float:
    """Z を 1e-6 m で丸めた深さの組を返す。"""
    key = round(float(z), Z_ROUND_DIGITS)
    if key == 0.0:
        return 0.0
    return key


def phase_label(filename: str) -> str:
    """ファイル名から段階名だけを取り出す。

    ``Calculation results Plate 掘削1 [Phase_4] ...`` なら ``掘削1``。
    パターンに合わないときは拡張子を除いたファイル名を返す。
    """
    match = _PHASE_RE.search(Path(filename).name)
    if match is None:
        return Path(filename).stem
    return match.group(1).strip()


def deduplicate_nodes(rows: Iterable[NodeRow]) -> list[NodeRow]:
    """同じ節点は最初の 1 行だけ残す。"""
    first: dict[str, NodeRow] = {}
    for row in rows:
        key = row.node.strip()
        if key not in first:
            first[key] = row
    return list(first.values())


def select_profile(
    rows: Iterable[NodeRow],
    y_target: float,
    tolerance: float,
) -> list[ProfilePoint]:
    """帯の中から、深さごとに目標 Y へ最も近い点だけを残す。

    手順は、節点の重複を 1 行へ落とす、``|Y - Y_target| <= tolerance`` で
    絞る、Z を 1e-6 m で丸めた組ごとに最近接点を残す、です。距離が同じ点は
    ``u_x`` を平均し、代表の Z はその点たちの Z の平均です。結果は Z の昇順です。
    """
    if tolerance < 0:
        raise ValueError("tolerance は 0 以上にしてください。")
    y_target = float(y_target)
    tolerance = float(tolerance)
    band = [
        row
        for row in deduplicate_nodes(rows)
        if abs(row.y - y_target) <= tolerance
    ]
    groups: dict[float, list[NodeRow]] = {}
    for row in band:
        groups.setdefault(depth_key(row.z), []).append(row)

    points: list[ProfilePoint] = []
    for group in groups.values():
        nearest = min(abs(row.y - y_target) for row in group)
        chosen = [
            row
            for row in group
            if abs(abs(row.y - y_target) - nearest) <= DISTANCE_TIE_M
        ]
        z_mean = sum(row.z for row in chosen) / len(chosen)
        ux_mean = sum(row.ux for row in chosen) / len(chosen)
        points.append(ProfilePoint(z=z_mean, ux=ux_mean))
    points.sort(key=lambda point: point.z)
    return points


def read_displacement_table(path: str | Path) -> list[NodeRow]:
    """UTF-8 BOM のタブ区切り全変位表を読む。節点の重複行は残す。"""
    table_path = Path(path)
    rows: list[NodeRow] = []
    with table_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"空のファイルです: {table_path}") from exc
        names = [cell.strip() for cell in header]
        index = {name: position for position, name in enumerate(names)}
        missing = [name for name in _REQUIRED_COLUMNS if name not in index]
        if missing:
            raise ValueError(f"列がありません {missing}: {table_path}")
        i_node = index["節点"]
        i_y = index["Y [m]"]
        i_z = index["Z [m]"]
        i_ux = index["u_x [m]"]
        width = max(i_node, i_y, i_z, i_ux) + 1
        for record in reader:
            if not record or all(not cell.strip() for cell in record):
                continue
            if len(record) < width:
                raise ValueError(f"列数が足りません: {table_path}")
            rows.append(
                NodeRow(
                    node=record[i_node].strip(),
                    y=float(record[i_y]),
                    z=float(record[i_z]),
                    ux=float(record[i_ux]),
                )
            )
    return rows


def list_displacement_files(folder: str | Path) -> list[Path]:
    """フォルダ内の変位表を、ファイル名の文字コード順で返す。"""
    folder_path = Path(folder)
    files = [
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    ]
    return sorted(files, key=lambda path: path.name)


def load_folder_series(
    folder: str | Path,
    y_target: float,
    tolerance: float,
) -> list[SeriesResult]:
    """フォルダの各ファイルを 1 系列にする。採用点 0 の系列も返す。"""
    series: list[SeriesResult] = []
    for path in list_displacement_files(folder):
        points = tuple(select_profile(read_displacement_table(path), y_target, tolerance))
        series.append(SeriesResult(label=phase_label(path.name), path=path, points=points))
    return series


def series_warnings(series: Sequence[SeriesResult]) -> list[str]:
    """採用点 0 件の系列について、描画を省く警告文を返す。"""
    return [
        f"警告: {item.label} は採用点が 0 件のため、この系列は描きません。"
        for item in series
        if item.n_points == 0
    ]


def _installed_family(names: Sequence[str]) -> str | None:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in names:
        if name in available:
            return name
    return None


def _register_noto_cjk() -> str | None:
    from matplotlib import font_manager

    for path in _NOTO_CJK_FILES:
        if not path.is_file():
            continue
        font_manager.fontManager.addfont(str(path))
        return str(font_manager.FontProperties(fname=str(path)).get_name())
    return None


def _import_japanize() -> str:
    try:
        import japanize_matplotlib  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "japanize-matplotlib"],
        )
        import japanize_matplotlib  # noqa: F401
    import matplotlib

    family = matplotlib.rcParams["font.family"]
    if isinstance(family, str):
        return family
    return str(family[0])


def configure_japanese_font() -> str:
    """凡例用の日本語フォントを選び、そのファミリー名を返す。

    macOS ではヒラギノ、次に Noto Sans CJK、どちらも無いときは
    ``japanize-matplotlib`` を入れます。
    """
    global _configured_font
    if _configured_font is not None:
        return _configured_font

    import matplotlib

    chosen = _installed_family(_HIRAGINO_FAMILIES)
    if chosen is None:
        chosen = _register_noto_cjk() or _installed_family(_NOTO_CJK_FAMILIES)
    if chosen is None:
        chosen = _import_japanize()
    else:
        matplotlib.rcParams["font.family"] = chosen
    matplotlib.rcParams["axes.unicode_minus"] = False
    _configured_font = chosen
    return chosen


def plot_profiles(
    series: Sequence[SeriesResult],
    output_path: str | Path | None = None,
    title: str | None = None,
):
    """系列を ``u_x [m]``–``Z [m]`` の折れ線にする。採用点 0 の系列は省く。"""
    configure_japanese_font()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 8.0))
    plotted = 0
    for item in series:
        if item.n_points == 0:
            continue
        plotted += 1
        ux = [point.ux for point in item.points]
        depth = [point.z for point in item.points]
        plot_kwargs: dict[str, object] = {"label": item.label, "linewidth": 1.6}
        color = PHASE_COLORS.get(item.label)
        if color is not None:
            plot_kwargs["color"] = color
        ax.plot(ux, depth, **plot_kwargs)
    ax.set_xlabel("u_x [m]")
    ax.set_ylabel("Z [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.05, y=0.03)
    if plotted:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.subplots_adjust(right=0.72)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150, bbox_inches="tight")
    return fig


def plot_folder(
    folder: str | Path,
    y_target: float,
    tolerance: float,
    output_path: str | Path | None = None,
    title: str | None = None,
):
    """フォルダ 1 つを 1 枚の図にする。戻り値は図と、空系列を含む系列一覧。"""
    series = load_folder_series(folder, y_target, tolerance)
    if title is None:
        title = f"{Path(folder).name}（Y = {y_target:g} m）"
    figure = plot_profiles(series, output_path=output_path, title=title)
    return figure, series
