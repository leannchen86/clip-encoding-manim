from manimlib import *
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distribution_metrics_shared import BG_COLOR, LABEL_COLOR, caption_text
from layout_verifier import LayoutVerifier


PANEL_STROKE = "#5c647d"
GOOD_COLOR = GREEN_C
BAD_COLOR = RED_C
NEUTRAL_COLOR = GREY_B
EMBED_COLORS = [BLUE_C, TEAL_C, YELLOW_D, ORANGE, PURPLE_B]


def make_pipeline_title(text, color):
    return Text(text, font="Helvetica Neue", color=color, font_size=27)


def make_embedding_tile(color, width=0.42, height=0.18):
    rect = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.035,
        stroke_color=WHITE,
        stroke_width=0.5,
        fill_color=color,
        fill_opacity=0.92,
    )
    return rect


def make_ordered_stack(pattern):
    tiles = VGroup(*[make_embedding_tile(color) for color in pattern])
    tiles.arrange(DOWN, buff=0.05)
    return tiles


def make_small_grid(pattern, cols=4, scale=0.9):
    tiles = VGroup(*[make_embedding_tile(color, width=0.36, height=0.16) for color in pattern])
    rows = []
    for i in range(0, len(tiles), cols):
        row = VGroup(*tiles[i:i + cols]).arrange(RIGHT, buff=0.05)
        rows.append(row)
    grid = VGroup(*rows).arrange(DOWN, buff=0.05)
    grid.scale(scale)
    return grid


class TruncateShufflePipelineScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        verifier = LayoutVerifier(scene_name="TruncateShufflePipelineScene")

        title = Text(
            "Truncate Before Shuffle vs. Shuffle Before Slice",
            font="Helvetica Neue",
            color=WHITE,
            font_size=31,
        )
        title.to_edge(UP, buff=0.26)

        source_pattern = [
            EMBED_COLORS[0], EMBED_COLORS[0], EMBED_COLORS[0], EMBED_COLORS[1],
            EMBED_COLORS[1], EMBED_COLORS[1], EMBED_COLORS[2], EMBED_COLORS[2],
            EMBED_COLORS[2], EMBED_COLORS[3], EMBED_COLORS[3], EMBED_COLORS[4],
            EMBED_COLORS[4], EMBED_COLORS[4],
        ]
        prefix_pattern = source_pattern[:8]
        shuffled_mixed = [
            EMBED_COLORS[2], EMBED_COLORS[4], EMBED_COLORS[0], EMBED_COLORS[3],
            EMBED_COLORS[1], EMBED_COLORS[4], EMBED_COLORS[2], EMBED_COLORS[0],
            EMBED_COLORS[3], EMBED_COLORS[1], EMBED_COLORS[4], EMBED_COLORS[2],
        ]
        truncated_prefix = prefix_pattern
        shuffled_slice = shuffled_mixed[:8]

        source_note_left = caption_text("ordered embeddings for one name", scale=0.31, color=LABEL_COLOR)
        source_note_right = caption_text("same raw cache, same order", scale=0.31, color=LABEL_COLOR)
        left_title = make_pipeline_title("truncate first", BAD_COLOR)
        right_title = make_pipeline_title("shuffle first", GOOD_COLOR)

        left_stack = make_ordered_stack(source_pattern).scale(0.95)
        right_stack = make_ordered_stack(source_pattern).scale(0.95)

        source_note_left.next_to(left_title, DOWN, buff=0.10)
        source_note_right.next_to(right_title, DOWN, buff=0.10)

        left_panel = VGroup(left_title, source_note_left, left_stack).arrange(DOWN, buff=0.12)
        right_panel = VGroup(right_title, source_note_right, right_stack).arrange(DOWN, buff=0.12)

        left_panel.move_to(LEFT * 3.45 + UP * 0.80)
        right_panel.move_to(RIGHT * 3.45 + UP * 0.80)

        left_prefix_box = SurroundingRectangle(
            VGroup(*left_stack[:8]),
            buff=0.05,
            stroke_color=BAD_COLOR,
            stroke_width=2.5,
        )
        left_prefix_label = caption_text("take first N", scale=0.33, color=BAD_COLOR)
        left_prefix_label.next_to(left_prefix_box, RIGHT, buff=0.16)

        right_shuffle_box = SurroundingRectangle(
            right_stack,
            buff=0.07,
            stroke_color=GOOD_COLOR,
            stroke_width=2.3,
        )
        right_shuffle_label = caption_text("shuffle full list first", scale=0.33, color=GOOD_COLOR)
        right_shuffle_label.next_to(right_shuffle_box, RIGHT, buff=0.14)

        left_arrow = Arrow(
            left_stack.get_bottom() + DOWN * 0.12,
            left_stack.get_bottom() + DOWN * 0.95,
            buff=0.0,
            stroke_color=BAD_COLOR,
            stroke_width=2.8,
        )
        right_arrow = Arrow(
            right_stack.get_bottom() + DOWN * 0.12,
            right_stack.get_bottom() + DOWN * 0.95,
            buff=0.0,
            stroke_color=GOOD_COLOR,
            stroke_width=2.8,
        )

        left_grid = make_small_grid(truncated_prefix, cols=4, scale=0.84)
        right_grid = make_small_grid(shuffled_slice, cols=4, scale=0.84)
        left_grid.move_to(left_stack.get_bottom() + DOWN * 1.42)
        right_grid.move_to(right_stack.get_bottom() + DOWN * 1.42)

        left_grid_label = caption_text("biased prefix only", scale=0.31, color=BAD_COLOR)
        right_grid_label = caption_text("same N after true shuffle", scale=0.31, color=GOOD_COLOR)
        left_grid_label.next_to(left_grid, UP, buff=0.12)
        right_grid_label.next_to(right_grid, UP, buff=0.12)

        left_result = RoundedRectangle(
            width=2.45,
            height=1.05,
            corner_radius=0.12,
            stroke_color=BAD_COLOR,
            stroke_width=2.0,
            fill_color=BAD_COLOR,
            fill_opacity=0.08,
        )
        left_result_text = VGroup(
            caption_text("biased slice", scale=0.32, color=BAD_COLOR),
            Text("~8.7x", font="Helvetica Neue", color=WHITE, font_size=28),
        ).arrange(DOWN, buff=0.08)
        left_result_group = VGroup(left_result, left_result_text)
        left_result_text.move_to(left_result.get_center())
        left_result_group.scale(0.82)
        left_result_group.move_to(LEFT * 3.45 + DOWN * 3.12)

        right_result = RoundedRectangle(
            width=2.45,
            height=1.05,
            corner_radius=0.12,
            stroke_color=GOOD_COLOR,
            stroke_width=2.0,
            fill_color=GOOD_COLOR,
            fill_opacity=0.08,
        )
        right_result_text = VGroup(
            caption_text("representative sample", scale=0.32, color=GOOD_COLOR),
            Text("~10.3x", font="Helvetica Neue", color=WHITE, font_size=28),
        ).arrange(DOWN, buff=0.08)
        right_result_group = VGroup(right_result, right_result_text)
        right_result_text.move_to(right_result.get_center())
        right_result_group.scale(0.82)
        right_result_group.move_to(RIGHT * 3.45 + DOWN * 3.15)

        left_to_result = Arrow(
            left_grid.get_bottom() + DOWN * 0.08,
            left_result_group.get_top() + UP * 0.05,
            buff=0.06,
            stroke_color=BAD_COLOR,
            stroke_width=2.6,
        )
        right_to_result = Arrow(
            right_grid.get_bottom() + DOWN * 0.08,
            right_result_group.get_top() + UP * 0.05,
            buff=0.06,
            stroke_color=GOOD_COLOR,
            stroke_width=2.6,
        )

        bottom_caption = caption_text(
            "A sampling bug can hide the real model behavior",
            scale=0.37,
            color=WHITE,
        )
        bottom_caption.to_edge(DOWN, buff=0.10)

        self.play(FadeIn(title, shift=DOWN * 0.10), run_time=0.6)
        self.play(
            FadeIn(left_title, shift=UP * 0.08),
            FadeIn(right_title, shift=UP * 0.08),
            run_time=0.55,
        )
        self.play(
            LaggedStart(*[FadeIn(tile, scale=0.5) for tile in left_stack], lag_ratio=0.03),
            LaggedStart(*[FadeIn(tile, scale=0.5) for tile in right_stack], lag_ratio=0.03),
            FadeIn(source_note_left, shift=UP * 0.06),
            FadeIn(source_note_right, shift=UP * 0.06),
            run_time=1.3,
        )

        self.play(
            ShowCreation(left_prefix_box),
            FadeIn(left_prefix_label, shift=LEFT * 0.05),
            run_time=0.6,
        )
        self.play(GrowArrow(left_arrow), run_time=0.35)
        self.play(
            LaggedStart(
                *[TransformFromCopy(tile, small) for tile, small in zip(left_stack[:8], left_grid)],
                lag_ratio=0.06,
            ),
            FadeIn(left_grid_label, shift=UP * 0.06),
            run_time=1.0,
        )

        self.play(
            ShowCreation(right_shuffle_box),
            FadeIn(right_shuffle_label, shift=LEFT * 0.05),
            run_time=0.6,
        )
        self.play(GrowArrow(right_arrow), run_time=0.35)
        self.play(
            LaggedStart(
                *[
                    TransformFromCopy(
                        right_stack[i % len(right_stack)],
                        right_grid[i],
                    )
                    for i in range(len(right_grid))
                ],
                lag_ratio=0.05,
            ),
            FadeIn(right_grid_label, shift=UP * 0.06),
            run_time=1.1,
        )

        self.play(
            GrowArrow(left_to_result),
            GrowArrow(right_to_result),
            FadeIn(left_result_group, shift=UP * 0.08),
            FadeIn(right_result_group, shift=UP * 0.08),
            run_time=0.8,
        )
        self.play(FadeIn(bottom_caption, shift=UP * 0.10), run_time=0.55)

        verifier.check_inside_frame("title", title, margin=0.08)
        verifier.check_inside_frame("left_panel", left_panel, margin=0.08)
        verifier.check_inside_frame("right_panel", right_panel, margin=0.08)
        verifier.check_inside_frame("source_note_left", source_note_left, margin=0.08)
        verifier.check_inside_frame("source_note_right", source_note_right, margin=0.08)
        verifier.check_inside_frame("left_grid", left_grid, margin=0.08)
        verifier.check_inside_frame("right_grid", right_grid, margin=0.08)
        verifier.check_inside_frame("left_grid_label", left_grid_label, margin=0.08)
        verifier.check_inside_frame("right_grid_label", right_grid_label, margin=0.08)
        verifier.check_inside_frame("left_result_group", left_result_group, margin=0.08)
        verifier.check_inside_frame("right_result_group", right_result_group, margin=0.08)
        verifier.check_inside_frame("bottom_caption", bottom_caption, margin=0.08)
        verifier.check_vertical_order("title", title, "left_panel", left_panel, min_gap=0.12)
        verifier.check_vertical_order("title", title, "right_panel", right_panel, min_gap=0.12)
        verifier.check_vertical_order("left_title", left_title, "source_note_left", source_note_left, min_gap=0.06)
        verifier.check_vertical_order("right_title", right_title, "source_note_right", source_note_right, min_gap=0.06)
        verifier.check_vertical_order("source_note_left", source_note_left, "left_stack", left_stack, min_gap=0.10)
        verifier.check_vertical_order("source_note_right", source_note_right, "right_stack", right_stack, min_gap=0.10)
        verifier.check_vertical_order("left_panel", left_panel, "left_grid", left_grid, min_gap=0.18)
        verifier.check_vertical_order("right_panel", right_panel, "right_grid", right_grid, min_gap=0.18)
        verifier.check_vertical_order("left_grid_label", left_grid_label, "left_grid", left_grid, min_gap=0.06)
        verifier.check_vertical_order("right_grid_label", right_grid_label, "right_grid", right_grid, min_gap=0.06)
        verifier.check_vertical_order("left_grid", left_grid, "left_result_group", left_result_group, min_gap=0.08)
        verifier.check_vertical_order("right_grid", right_grid, "right_result_group", right_result_group, min_gap=0.08)
        verifier.check_vertical_order("left_result_group", left_result_group, "bottom_caption", bottom_caption, min_gap=0.12)
        verifier.check_vertical_order("right_result_group", right_result_group, "bottom_caption", bottom_caption, min_gap=0.12)
        verifier.check_no_overlap("left_grid_label", left_grid_label, "left_result_group", left_result_group, min_gap=0.08)
        verifier.check_no_overlap("right_grid_label", right_grid_label, "right_result_group", right_result_group, min_gap=0.08)
        for i, (start_tile, end_tile) in enumerate(zip(left_stack[:8], left_grid)):
            verifier.check_motion_path_clearance(
                "source_note_left",
                source_note_left,
                f"left_tile_copy[{i}]",
                start_tile,
                end_tile,
                min_gap=0.04,
            )
        for i, end_tile in enumerate(right_grid):
            verifier.check_motion_path_clearance(
                "source_note_right",
                source_note_right,
                f"right_tile_copy[{i}]",
                right_stack[i % len(right_stack)],
                end_tile,
                min_gap=0.04,
            )
        verifier.check_min_horizontal_gap("left_panel", left_panel, "right_panel", right_panel, min_gap=0.80)
        verifier.check_min_horizontal_gap("left_grid", left_grid, "right_grid", right_grid, min_gap=0.70)
        verifier.check_min_horizontal_gap("left_result_group", left_result_group, "right_result_group", right_result_group, min_gap=0.85)
        verifier.check_no_overlap("left_prefix_label", left_prefix_label, "right_panel", right_panel, min_gap=0.08)
        verifier.check_no_overlap("right_shuffle_label", right_shuffle_label, "title", title, min_gap=0.06)
        verifier.assert_ok()

        self.wait(1.5)
