from manimlib import *
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distribution_metrics_shared import BG_COLOR, LABEL_COLOR, caption_text
from layout_verifier import LayoutVerifier


PANEL_STROKE = "#5c647d"
MINI_COLOR = ORANGE
FULL_COLOR = GREEN_C
DATA_COLORS = [BLUE_C, TEAL_C, PURPLE_B, YELLOW_D]


def make_panel_title(text, color):
    return Text(text, font="Helvetica Neue", color=color, font_size=28)


def make_dataset_grid(scale=1.0):
    cells = VGroup()
    pattern = [DATA_COLORS[0], DATA_COLORS[1], DATA_COLORS[2], DATA_COLORS[3]] * 3
    for color in pattern:
        rect = RoundedRectangle(
            width=0.34,
            height=0.24,
            corner_radius=0.04,
            stroke_color=WHITE,
            stroke_width=0.45,
            fill_color=color,
            fill_opacity=0.90,
        )
        cells.add(rect)
    rows = VGroup()
    for i in range(0, len(cells), 4):
        rows.add(VGroup(*cells[i:i + 4]).arrange(RIGHT, buff=0.06))
    grid = VGroup(*rows).arrange(DOWN, buff=0.06)
    grid.scale(scale)
    return grid


def make_contour_plot(color, path_points, smooth=True):
    outer = Ellipse(width=3.0, height=2.0, stroke_color=PANEL_STROKE, stroke_opacity=0.35, stroke_width=1.4)
    mid = Ellipse(width=2.1, height=1.35, stroke_color=PANEL_STROKE, stroke_opacity=0.35, stroke_width=1.4)
    inner = Ellipse(width=1.15, height=0.70, stroke_color=PANEL_STROKE, stroke_opacity=0.38, stroke_width=1.4)
    goal = Dot(radius=0.06, color=color)
    goal.move_to(RIGHT * 0.40 + DOWN * 0.32)

    path = VMobject()
    mapped = [LEFT * x + UP * y for x, y in path_points]
    if smooth:
        path.set_points_smoothly(mapped)
    else:
        path.set_points_as_corners(mapped)
    path.set_stroke(color=color, width=3.0, opacity=0.95)

    dots = VGroup(*[
        Dot(point, radius=0.045, color=color).set_opacity(0.9)
        for point in mapped
    ])
    return VGroup(outer, mid, inner, path, dots, goal), path, dots, goal


def make_result_chip(label, stat_text, color):
    box = RoundedRectangle(
        width=1.90,
        height=0.84,
        corner_radius=0.12,
        stroke_color=color,
        stroke_width=1.8,
        fill_color=color,
        fill_opacity=0.08,
    )
    top = caption_text(label, scale=0.30, color=color)
    stat = Text(stat_text, font="Helvetica Neue", color=WHITE, font_size=24)
    text_group = VGroup(top, stat).arrange(DOWN, buff=0.06)
    text_group.move_to(box.get_center())
    return VGroup(box, text_group)


class FullBatchVsMiniBatchScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        verifier = LayoutVerifier(scene_name="FullBatchVsMiniBatchScene")

        title = Text(
            "Full Batch vs. Mini-Batch Gradient Noise",
            font="Helvetica Neue",
            color=WHITE,
            font_size=31,
        )
        title.to_edge(UP, buff=0.28)
        title_note = caption_text(
            "Toy 2D loss slice over parameter space",
            scale=0.31,
            color=LABEL_COLOR,
        )
        title_note.next_to(title, DOWN, buff=0.10)

        left_title = make_panel_title("mini-batch (4096)", MINI_COLOR)
        right_title = make_panel_title("full batch (690K)", FULL_COLOR)
        left_title.move_to(LEFT * 3.45 + UP * 2.35)
        right_title.move_to(RIGHT * 3.45 + UP * 2.35)

        left_grid = make_dataset_grid(scale=0.92)
        right_grid = make_dataset_grid(scale=0.92)
        left_grid.move_to(LEFT * 3.45 + UP * 1.42)
        right_grid.move_to(RIGHT * 3.45 + UP * 1.42)

        mini_box = SurroundingRectangle(VGroup(left_grid[0][0], left_grid[0][1], left_grid[1][0], left_grid[1][1]), buff=0.08, stroke_color=MINI_COLOR, stroke_width=2.4)
        full_box = SurroundingRectangle(right_grid, buff=0.08, stroke_color=FULL_COLOR, stroke_width=2.4)
        mini_label = caption_text("one noisy slice", scale=0.31, color=MINI_COLOR)
        full_label = caption_text("all embeddings at once", scale=0.31, color=FULL_COLOR)
        mini_label.next_to(mini_box, DOWN, buff=0.12)
        full_label.next_to(full_box, DOWN, buff=0.12)

        mini_points = [
            (1.35, 1.05), (1.05, 0.82), (1.20, 0.58), (0.82, 0.42),
            (0.98, 0.16), (0.62, 0.08), (0.74, -0.16), (0.42, -0.06),
            (0.26, -0.28), (0.02, -0.20),
        ]
        full_points = [
            (1.35, 1.05), (1.05, 0.78), (0.78, 0.52), (0.55, 0.26),
            (0.28, 0.02), (0.04, -0.20),
        ]

        left_plot, left_path, left_dots, left_goal = make_contour_plot(MINI_COLOR, mini_points, smooth=False)
        right_plot, right_path, right_dots, right_goal = make_contour_plot(FULL_COLOR, full_points, smooth=True)
        left_plot.move_to(LEFT * 3.45 + DOWN * 0.85)
        right_plot.move_to(RIGHT * 3.45 + DOWN * 0.85)

        left_arrow = Arrow(
            left_grid.get_bottom() + DOWN * 0.08,
            left_dots[0].get_center() + UP * 0.10,
            buff=0.0,
            stroke_color=MINI_COLOR,
            stroke_width=2.6,
        )
        right_arrow = Arrow(
            right_grid.get_bottom() + DOWN * 0.08,
            right_dots[0].get_center() + UP * 0.10,
            buff=0.0,
            stroke_color=FULL_COLOR,
            stroke_width=2.6,
        )
        left_arrow_label = caption_text("gradient updates", scale=0.27, color=MINI_COLOR)
        right_arrow_label = caption_text("gradient updates", scale=0.27, color=FULL_COLOR)
        left_arrow_label.next_to(left_arrow, LEFT, buff=0.10)
        right_arrow_label.next_to(right_arrow, RIGHT, buff=0.10)

        left_space_tag = caption_text("toy 2D loss slice", scale=0.28, color=LABEL_COLOR)
        right_space_tag = caption_text("toy 2D loss slice", scale=0.28, color=LABEL_COLOR)
        left_space_tag.next_to(left_plot, UP, buff=0.12)
        right_space_tag.next_to(right_plot, UP, buff=0.12)

        left_goal_label = caption_text("lower loss", scale=0.24, color=MINI_COLOR)
        right_goal_label = caption_text("lower loss", scale=0.24, color=FULL_COLOR)
        left_goal_label.next_to(left_goal, RIGHT, buff=0.10)
        right_goal_label.next_to(right_goal, RIGHT, buff=0.10)

        left_plot_label = caption_text("more jitter in each update", scale=0.31, color=MINI_COLOR)
        right_plot_label = caption_text("cleaner gradient direction", scale=0.31, color=FULL_COLOR)
        left_plot_label.next_to(left_plot, DOWN, buff=0.12)
        right_plot_label.next_to(right_plot, DOWN, buff=0.12)

        left_result = make_result_chip("mini-batch result", "~9.9x", MINI_COLOR)
        right_result = make_result_chip("full-batch result", "~10.2x", FULL_COLOR)
        left_result.move_to(LEFT * 3.45 + DOWN * 2.92)
        right_result.move_to(RIGHT * 3.45 + DOWN * 2.92)

        memory_note = caption_text("All embeddings fit in memory", scale=0.33, color=WHITE)
        memory_note.move_to(DOWN * 2.25)

        footer = caption_text(
            "When the problem is already clean, extra gradient noise may not help",
            scale=0.39,
            color=WHITE,
        )
        footer.to_edge(DOWN, buff=0.10)

        self.play(
            FadeIn(title, shift=DOWN * 0.10),
            FadeIn(title_note, shift=DOWN * 0.06),
            run_time=0.65,
        )
        self.play(
            FadeIn(left_title, shift=UP * 0.08),
            FadeIn(right_title, shift=UP * 0.08),
            run_time=0.55,
        )
        self.play(
            LaggedStart(*[FadeIn(cell, scale=0.55) for cell in left_grid], lag_ratio=0.03),
            LaggedStart(*[FadeIn(cell, scale=0.55) for cell in right_grid], lag_ratio=0.03),
            run_time=1.1,
        )
        self.play(
            ShowCreation(mini_box),
            ShowCreation(full_box),
            FadeIn(mini_label, shift=UP * 0.05),
            FadeIn(full_label, shift=UP * 0.05),
            run_time=0.75,
        )
        self.play(
            GrowArrow(left_arrow),
            GrowArrow(right_arrow),
            FadeIn(left_arrow_label, shift=UP * 0.04),
            FadeIn(right_arrow_label, shift=UP * 0.04),
            run_time=0.55,
        )
        self.play(
            FadeIn(left_plot[:3], scale=0.96),
            FadeIn(right_plot[:3], scale=0.96),
            FadeIn(left_space_tag, shift=UP * 0.05),
            FadeIn(right_space_tag, shift=UP * 0.05),
            run_time=0.65,
        )
        self.play(
            ShowCreation(left_path),
            FadeIn(left_dots),
            FadeIn(left_goal, scale=0.85),
            FadeIn(left_goal_label, shift=RIGHT * 0.04),
            FadeIn(left_plot_label, shift=UP * 0.05),
            run_time=1.0,
        )
        self.play(
            ShowCreation(right_path),
            FadeIn(right_dots),
            FadeIn(right_goal, scale=0.85),
            FadeIn(right_goal_label, shift=RIGHT * 0.04),
            FadeIn(right_plot_label, shift=UP * 0.05),
            run_time=0.9,
        )
        self.play(
            FadeIn(memory_note, shift=UP * 0.08),
            FadeIn(left_result, shift=UP * 0.08),
            FadeIn(right_result, shift=UP * 0.08),
            run_time=0.7,
        )
        self.play(FadeIn(footer, shift=UP * 0.10), run_time=0.55)

        verifier.check_inside_frame("title", title, margin=0.08)
        verifier.check_inside_frame("title_note", title_note, margin=0.08)
        verifier.check_inside_frame("left_title", left_title, margin=0.08)
        verifier.check_inside_frame("right_title", right_title, margin=0.08)
        verifier.check_inside_frame("left_grid", left_grid, margin=0.08)
        verifier.check_inside_frame("right_grid", right_grid, margin=0.08)
        verifier.check_inside_frame("left_plot", left_plot, margin=0.08)
        verifier.check_inside_frame("right_plot", right_plot, margin=0.08)
        verifier.check_inside_frame("left_result", left_result, margin=0.08)
        verifier.check_inside_frame("right_result", right_result, margin=0.08)
        verifier.check_inside_frame("footer", footer, margin=0.08)
        verifier.check_vertical_order("title", title, "title_note", title_note, min_gap=0.08)
        verifier.check_vertical_order("title_note", title_note, "left_title", left_title, min_gap=0.10)
        verifier.check_vertical_order("title_note", title_note, "right_title", right_title, min_gap=0.10)
        verifier.check_vertical_order("left_grid", left_grid, "left_plot", left_plot, min_gap=0.14)
        verifier.check_vertical_order("right_grid", right_grid, "right_plot", right_plot, min_gap=0.14)
        verifier.check_vertical_order("left_plot_label", left_plot_label, "left_result", left_result, min_gap=0.14)
        verifier.check_vertical_order("right_plot_label", right_plot_label, "right_result", right_result, min_gap=0.14)
        verifier.check_vertical_order("memory_note", memory_note, "footer", footer, min_gap=0.14)
        verifier.check_min_horizontal_gap("left_plot", left_plot, "right_plot", right_plot, min_gap=1.10)
        verifier.assert_ok()

        self.wait(1.5)
