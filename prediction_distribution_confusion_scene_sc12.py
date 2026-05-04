from manimlib import *
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distribution_metrics_shared import BG_COLOR, LABEL_COLOR, caption_text
from layout_verifier import LayoutVerifier


PANEL_STROKE = "#5c647d"
NAMES = ["Todd", "Kyle", "Stuart", "Bob", "Pierre", "David"]
NAME_COLORS = [YELLOW_D, TEAL_C, BLUE_C, MAROON_B, PURPLE_B, GREY_B]
TRUE_SHARE = 0.7
PREDICTED_SHARE = [7.0, 4.5, 3.9, 3.2, 2.7, 0.8]

CONFUSION = [
    [0.43, 0.15, 0.10, 0.09, 0.08, 0.15],
    [0.24, 0.39, 0.11, 0.08, 0.07, 0.11],
    [0.21, 0.15, 0.38, 0.09, 0.07, 0.10],
    [0.19, 0.11, 0.08, 0.45, 0.09, 0.08],
    [0.17, 0.10, 0.08, 0.10, 0.46, 0.09],
    [0.25, 0.12, 0.09, 0.08, 0.06, 0.40],
]


def make_panel_title(text, color=WHITE):
    return caption_text(text, scale=0.46, color=color)


def make_name_card(name, color):
    card = RoundedRectangle(
        width=1.25,
        height=0.72,
        corner_radius=0.12,
        stroke_color=PANEL_STROKE,
        stroke_width=1.4,
        fill_color=color,
        fill_opacity=0.18,
    )
    label = Text(name, font="Helvetica Neue", color=color, font_size=22)
    pct = Text("0.7%", font="Helvetica Neue", color=WHITE, font_size=16)
    pct.set_opacity(0.9)
    text_group = VGroup(label, pct)
    text_group.arrange(DOWN, buff=0.08)
    text_group.move_to(card.get_center())
    return VGroup(card, text_group)


def make_true_distribution_panel():
    title = make_panel_title("True distribution")
    subtitle = caption_text(
        "Balanced 138-way validation set",
        scale=0.30,
        color=LABEL_COLOR,
    )
    cards = VGroup(*[
        make_name_card(name, color)
        for name, color in zip(NAMES, NAME_COLORS)
    ])
    left_col = VGroup(*cards[:3]).arrange(DOWN, buff=0.18)
    right_col = VGroup(*cards[3:]).arrange(DOWN, buff=0.18)
    cards_grid = VGroup(left_col, right_col).arrange(RIGHT, buff=0.18)
    cards_grid.scale(0.95)

    footer = caption_text(
        "Each name appears about equally often",
        scale=0.32,
        color=WHITE,
    )
    panel = VGroup(title, subtitle, cards_grid, footer)
    panel.arrange(DOWN, buff=0.16)
    panel.cards = cards
    panel.cards_grid = cards_grid
    panel.title = title
    panel.subtitle = subtitle
    panel.footer = footer
    return panel


def make_bar_chart_panel(values):
    title = make_panel_title("Predicted distribution")
    subtitle = caption_text(
        "Share of top-1 predictions",
        scale=0.30,
        color=LABEL_COLOR,
    )

    axes = Axes(
        x_range=[0, 7, 1],
        y_range=[0, 8, 2],
        width=5.1,
        height=2.8,
        axis_config=dict(
            stroke_color=PANEL_STROKE,
            stroke_width=1.6,
            include_ticks=False,
            include_tip=False,
        ),
    )
    axes_labels = VGroup()
    bars = VGroup()
    value_labels = VGroup()

    x_slots = [0.8 + i for i in range(len(NAMES))]
    bar_width = 0.54

    for x, name, value, color in zip(x_slots, NAMES, values, NAME_COLORS):
        bottom_left = axes.c2p(x - bar_width / 2, 0)
        top_right = axes.c2p(x + bar_width / 2, value)
        rect = Rectangle(
            width=abs(top_right[0] - bottom_left[0]),
            height=abs(top_right[1] - bottom_left[1]),
            stroke_width=0,
            fill_color=color,
            fill_opacity=0.88,
        )
        rect.move_to((bottom_left + top_right) / 2)
        bars.add(rect)

        label = Text(name, font="Helvetica Neue", color=LABEL_COLOR, font_size=17)
        label.rotate(PI / 10)
        label.next_to(axes.c2p(x, 0), DOWN, buff=0.18)
        axes_labels.add(label)

        value_label = Text(
            f"{value:.1f}%",
            font="Helvetica Neue",
            color=WHITE,
            font_size=16,
        )
        value_label.next_to(rect, UP, buff=0.09)
        value_labels.add(value_label)

    y_ticks = VGroup()
    for y in [0, 2, 4, 6, 8]:
        tick_label = Text(
            f"{y}%",
            font="Helvetica Neue",
            color=LABEL_COLOR,
            font_size=15,
        )
        tick_label.next_to(axes.c2p(0, y), LEFT, buff=0.12)
        y_ticks.add(tick_label)

    panel = VGroup(title, subtitle, axes, y_ticks, bars, axes_labels, value_labels)
    title.next_to(axes, UP, buff=0.56)
    subtitle.next_to(title, DOWN, buff=0.10)
    panel.title = title
    panel.subtitle = subtitle
    panel.axes = axes
    panel.y_ticks = y_ticks
    panel.bars = bars
    panel.axes_labels = axes_labels
    panel.value_labels = value_labels
    return panel


def make_confusion_matrix(names, matrix):
    title = make_panel_title("Where the wrong guesses go")
    subtitle = caption_text(
        "Rows = true name, columns = predicted name",
        scale=0.28,
        color=LABEL_COLOR,
    )

    cell = 0.48
    gap = 0.06
    grid = VGroup()
    column_groups = [VGroup() for _ in names]
    row_groups = [VGroup() for _ in names]

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            is_diag = i == j
            fill_color = GREEN_C if is_diag else RED_C
            opacity = 0.18 + 0.58 * min(1.0, value / (0.46 if is_diag else 0.25))
            box = RoundedRectangle(
                width=cell,
                height=cell,
                corner_radius=0.05,
                stroke_color=PANEL_STROKE,
                stroke_width=1.0,
                fill_color=fill_color,
                fill_opacity=opacity,
            )
            box.move_to(
                RIGHT * (j * (cell + gap)) + DOWN * (i * (cell + gap))
            )
            pct = Text(
                f"{int(round(value * 100))}%",
                font="Helvetica Neue",
                color=WHITE,
                font_size=14,
            )
            pct.move_to(box.get_center())
            group = VGroup(box, pct)
            grid.add(group)
            column_groups[j].add(group)
            row_groups[i].add(group)

    grid.move_to(ORIGIN)

    row_labels = VGroup()
    col_labels = VGroup()
    for i, name in enumerate(names):
        row_label = Text(name, font="Helvetica Neue", color=LABEL_COLOR, font_size=17)
        row_label.next_to(row_groups[i], LEFT, buff=0.15)
        row_labels.add(row_label)

        col_label = Text(name, font="Helvetica Neue", color=LABEL_COLOR, font_size=17)
        col_label.rotate(PI / 10)
        col_label.next_to(column_groups[i], UP, buff=0.14)
        col_labels.add(col_label)

    row_header = Text("true", font="Helvetica Neue", color=WHITE, font_size=16)
    row_header.next_to(row_labels, LEFT, buff=0.15)
    row_header.rotate(PI / 2)

    col_header = Text("predicted", font="Helvetica Neue", color=WHITE, font_size=16)
    col_header.next_to(col_labels, UP, buff=0.12)

    matrix_group = VGroup(
        title,
        subtitle,
        grid,
        row_labels,
        col_labels,
        row_header,
        col_header,
    )
    title.next_to(grid, UP, buff=1.0)
    subtitle.next_to(title, DOWN, buff=0.10)
    matrix_group.grid = grid
    matrix_group.column_groups = column_groups
    matrix_group.row_groups = row_groups
    matrix_group.row_labels = row_labels
    matrix_group.col_labels = col_labels
    matrix_group.title = title
    matrix_group.subtitle = subtitle
    matrix_group.row_header = row_header
    matrix_group.col_header = col_header
    return matrix_group


class PredictionDistributionConfusionScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        verifier = LayoutVerifier(scene_name="PredictionDistributionConfusionScene")

        title = Text(
            "Prediction Distribution and Over-Predicted Names",
            font="Helvetica Neue",
            color=WHITE,
            font_size=32,
        )
        title.to_edge(UP, buff=0.28)

        true_panel = make_true_distribution_panel()
        predicted_panel = make_bar_chart_panel(PREDICTED_SHARE)

        true_panel.scale(0.84)
        predicted_panel.scale(0.80)

        true_panel.move_to(LEFT * 3.35 + UP * 1.85)
        predicted_panel.move_to(RIGHT * 3.05 + UP * 1.72)

        main_caption = caption_text(
            "Accuracy alone does not show where wrong guesses go",
            scale=0.39,
            color=WHITE,
        )
        main_caption.move_to(DOWN * 0.15)

        todd_bar = predicted_panel.bars[0]
        todd_label = predicted_panel.axes_labels[0]
        todd_highlight = SurroundingRectangle(
            VGroup(todd_bar, todd_label, predicted_panel.value_labels[0]),
            buff=0.10,
            stroke_color=YELLOW,
            stroke_width=2.4,
        )

        todd_note = caption_text(
            "Todd ~ 7.0% predicted\nExpected ~ 0.7%",
            scale=0.34,
            color=YELLOW,
        )
        todd_note.next_to(predicted_panel.bars[0], RIGHT, buff=0.48)
        todd_note.shift(DOWN * 0.05)

        todd_arrow = Arrow(
            todd_note.get_left() + RIGHT * 0.05,
            predicted_panel.value_labels[0].get_right() + LEFT * 0.03,
            buff=0.10,
            stroke_color=YELLOW,
            stroke_width=3,
        )

        matrix = make_confusion_matrix(NAMES, CONFUSION)
        matrix.scale(0.80)
        matrix.move_to(DOWN * 1.72 + RIGHT * 0.12)

        todd_column_glow = SurroundingRectangle(
            VGroup(matrix.column_groups[0], matrix.col_labels[0]),
            buff=0.10,
            stroke_color=YELLOW,
            stroke_width=2.6,
        )
        kyle_column_glow = SurroundingRectangle(
            VGroup(matrix.column_groups[1], matrix.col_labels[1]),
            buff=0.10,
            stroke_color=TEAL_C,
            stroke_width=2.0,
        )
        kyle_column_glow.set_stroke(opacity=0.7)

        matrix_note = caption_text(
            "A few names absorb many wrong guesses",
            scale=0.36,
            color=WHITE,
        )
        matrix_note.next_to(matrix.grid, DOWN, buff=0.18)

        true_vs_pred_link = Arrow(
            start=true_panel.cards[0].get_right() + RIGHT * 0.10,
            end=predicted_panel.bars[0].get_left() + LEFT * 0.08,
            buff=0.10,
            stroke_color=GREY_B,
            stroke_width=2.0,
        )
        true_vs_pred_link.set_opacity(0.45)

        self.play(FadeIn(title, shift=DOWN * 0.10), run_time=0.65)

        self.play(
            FadeIn(true_panel.title, shift=UP * 0.12),
            FadeIn(true_panel.subtitle, shift=UP * 0.12),
            LaggedStart(*[FadeIn(card, scale=0.92) for card in true_panel.cards], lag_ratio=0.08),
            FadeIn(true_panel.footer, shift=UP * 0.08),
            run_time=1.3,
        )

        self.play(
            FadeIn(predicted_panel.title, shift=UP * 0.12),
            FadeIn(predicted_panel.subtitle, shift=UP * 0.12),
            ShowCreation(predicted_panel.axes),
            FadeIn(predicted_panel.y_ticks, shift=RIGHT * 0.05),
            FadeIn(predicted_panel.axes_labels, shift=UP * 0.08),
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in predicted_panel.bars], lag_ratio=0.08),
            FadeIn(predicted_panel.value_labels, shift=UP * 0.05),
            run_time=1.5,
        )
        self.play(ShowCreation(true_vs_pred_link), FadeIn(main_caption, shift=UP * 0.08), run_time=0.75)
        self.play(
            ShowCreation(todd_highlight),
            GrowArrow(todd_arrow),
            FadeIn(todd_note, shift=LEFT * 0.08),
            run_time=0.85,
        )

        self.play(
            FadeOut(true_vs_pred_link),
            FadeOut(main_caption),
            run_time=0.35,
        )

        self.play(
            FadeIn(matrix.title, shift=UP * 0.12),
            FadeIn(matrix.subtitle, shift=UP * 0.12),
            LaggedStart(*[FadeIn(cell, scale=0.9) for cell in matrix.grid], lag_ratio=0.015),
            FadeIn(matrix.row_labels, shift=RIGHT * 0.06),
            FadeIn(matrix.col_labels, shift=DOWN * 0.06),
            FadeIn(matrix.row_header, shift=RIGHT * 0.05),
            FadeIn(matrix.col_header, shift=DOWN * 0.05),
            run_time=1.6,
        )
        self.play(
            ShowCreation(todd_column_glow),
            ShowCreation(kyle_column_glow),
            FadeIn(matrix_note, shift=UP * 0.08),
            run_time=0.85,
        )

        verifier.check_inside_frame("title", title, margin=0.08)
        verifier.check_inside_frame("true_panel", true_panel, margin=0.08)
        verifier.check_inside_frame("predicted_panel", predicted_panel, margin=0.08)
        verifier.check_inside_frame("todd_note", todd_note, margin=0.08)
        verifier.check_inside_frame("matrix", matrix, margin=0.06)
        verifier.check_inside_frame("matrix_note", matrix_note, margin=0.08)
        verifier.check_vertical_order("title", title, "true_panel", true_panel, min_gap=0.12)
        verifier.check_vertical_order("title", title, "predicted_panel", predicted_panel, min_gap=0.12)
        verifier.check_vertical_order("true_panel", true_panel, "matrix", matrix, min_gap=0.16)
        verifier.check_vertical_order("predicted_panel", predicted_panel, "matrix", matrix, min_gap=0.10)
        verifier.check_vertical_order("matrix", matrix, "matrix_note", matrix_note, min_gap=0.10)
        verifier.check_min_horizontal_gap("true_panel", true_panel, "predicted_panel", predicted_panel, min_gap=0.22)
        verifier.check_no_overlap("todd_note", todd_note, "title", title, min_gap=0.10)
        verifier.check_no_overlap("todd_note", todd_note, "predicted_panel", predicted_panel.title, min_gap=0.08)
        verifier.check_no_overlap("todd_note", todd_note, "matrix_title", matrix.title, min_gap=0.08)
        verifier.assert_ok()

        self.wait(1.5)
