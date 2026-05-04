from manimlib import *
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distribution_metrics_shared import BG_COLOR, LABEL_COLOR, caption_text
from layout_verifier import LayoutVerifier


PANEL_STROKE = "#5c647d"
WIDE_COLOR = RED_C
BN_COLOR = GREEN_C
INPUT_COLOR = BLUE_B
NEUTRAL_COLOR = GREY_B


def make_panel_title(text, color):
    return Text(text, font="Helvetica Neue", color=color, font_size=28)


def make_arch_label(text):
    return caption_text(text, scale=0.30, color=LABEL_COLOR)


def make_layer_caption(text, color):
    return caption_text(text, scale=0.27, color=color)


def make_result_card(title, stat_text, color):
    box = RoundedRectangle(
        width=2.35,
        height=1.00,
        corner_radius=0.14,
        stroke_color=color,
        stroke_width=2.0,
        fill_color=color,
        fill_opacity=0.08,
    )
    title_mob = caption_text(title, scale=0.31, color=color)
    stat = Text(stat_text, font="Helvetica Neue", color=WHITE, font_size=26)
    group = VGroup(title_mob, stat).arrange(DOWN, buff=0.08)
    group.move_to(box.get_center())
    return VGroup(box, group)


def make_neuron_column(count, color, radius=0.08, buff=0.18, fill_opacity=0.10):
    dots = VGroup(*[
        Dot(radius=radius, color=color).set_fill(color, opacity=fill_opacity)
        for _ in range(count)
    ])
    dots.arrange(DOWN, buff=buff)
    dots.set_stroke(color, width=1.2, opacity=0.75)
    return dots


def connect_layers(left_column, right_column, color, opacity=0.14, width=1.0):
    lines = VGroup()
    for left_dot in left_column:
        for right_dot in right_column:
            lines.add(
                Line(
                    left_dot.get_center(),
                    right_dot.get_center(),
                    stroke_color=color,
                    stroke_width=width,
                    stroke_opacity=opacity,
                )
            )
    return lines


def pulse_lines(lines, color):
    flashes = []
    sample = list(lines)
    stride = max(1, len(sample) // 8)
    for line in sample[::stride]:
        flash = line.copy()
        flash.set_stroke(color=color, width=3.0, opacity=0.9)
        flashes.append(ShowPassingFlash(flash, time_width=0.5))
    return LaggedStart(*flashes, lag_ratio=0.06)


def make_bn_block(label="BN"):
    def make_bar(width):
        bar = RoundedRectangle(
            width=width,
            height=0.07,
            corner_radius=0.025,
            stroke_width=0,
            fill_color=BN_COLOR,
            fill_opacity=0.90,
        )
        return bar

    box = RoundedRectangle(
        width=0.90,
        height=1.05,
        corner_radius=0.10,
        stroke_color=BN_COLOR,
        stroke_width=1.8,
        fill_color=BN_COLOR,
        fill_opacity=0.10,
    )

    left_bars = VGroup(
        make_bar(0.12),
        make_bar(0.24),
        make_bar(0.16),
    ).arrange(DOWN, buff=0.07, aligned_edge=LEFT)
    right_bars = VGroup(
        make_bar(0.18),
        make_bar(0.18),
        make_bar(0.18),
    ).arrange(DOWN, buff=0.07, aligned_edge=LEFT)

    inner_arrow = Arrow(
        LEFT * 0.08,
        RIGHT * 0.08,
        buff=0.0,
        stroke_color=BN_COLOR,
        stroke_width=1.8,
    )
    inner_arrow.set_opacity(0.55)

    icon = VGroup(left_bars, inner_arrow, right_bars).arrange(RIGHT, buff=0.08, aligned_edge=ORIGIN)
    icon.move_to(box.get_center() + UP * 0.08)
    left_bars.set_fill(BN_COLOR, opacity=0.85)
    right_bars.set_fill(BN_COLOR, opacity=0.22)

    text = Text(label, font="Helvetica Neue", color=BN_COLOR, font_size=16)
    text.move_to(box.get_center() + DOWN * 0.30)

    group = VGroup(box, icon, text)
    group.box = box
    group.left_bars = left_bars
    group.inner_arrow = inner_arrow
    group.right_bars = right_bars
    group.label = text
    return group


class BatchNormBeatsWidthScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        verifier = LayoutVerifier(scene_name="BatchNormBeatsWidthScene")

        title = Text(
            "BatchNorm Beats Width",
            font="Helvetica Neue",
            color=WHITE,
            font_size=32,
        )
        title.to_edge(UP, buff=0.28)

        left_title = make_panel_title("wide MLP, no BatchNorm", WIDE_COLOR)
        right_title = make_panel_title("narrow MLP, with BatchNorm", BN_COLOR)
        left_arch = make_arch_label("768 -> 1024 -> 138")
        right_arch = make_arch_label("768 -> 256 -> BN -> 128 -> BN -> 138")

        left_header = VGroup(left_title, left_arch).arrange(DOWN, buff=0.10)
        right_header = VGroup(right_title, right_arch).arrange(DOWN, buff=0.10)
        left_header.move_to(LEFT * 3.35 + UP * 2.55)
        right_header.move_to(RIGHT * 3.00 + UP * 2.55)

        # Left network: wide hidden layer, no normalization.
        left_input = make_neuron_column(5, INPUT_COLOR, radius=0.075, buff=0.22, fill_opacity=0.14)
        left_hidden = make_neuron_column(9, WIDE_COLOR, radius=0.072, buff=0.12, fill_opacity=0.08)
        left_output = make_neuron_column(4, NEUTRAL_COLOR, radius=0.070, buff=0.22, fill_opacity=0.08)
        left_input.move_to(LEFT * 4.65 + UP * 0.35)
        left_hidden.move_to(LEFT * 3.50 + UP * 0.35)
        left_output.move_to(LEFT * 2.20 + UP * 0.35)

        left_in_to_hidden = connect_layers(left_input, left_hidden, WIDE_COLOR, opacity=0.13, width=0.9)
        left_hidden_to_out = connect_layers(left_hidden, left_output, WIDE_COLOR, opacity=0.12, width=0.9)

        left_input_label = make_layer_caption("same frozen\nCLIP features", INPUT_COLOR)
        left_hidden_label = make_layer_caption("1024 hidden units", WIDE_COLOR)
        left_output_label = make_layer_caption("138 names", LABEL_COLOR)
        left_input_label.next_to(left_input, UP, buff=0.18)
        left_hidden_label.next_to(left_hidden, DOWN, buff=0.14)
        left_output_label.next_to(left_output, DOWN, buff=0.14)

        left_takeaway = make_layer_caption("some channels dominate,\nothers fade out", WIDE_COLOR)
        left_takeaway.move_to(LEFT * 3.45 + DOWN * 1.65)

        # Right network: narrower architecture with BatchNorm blocks between layers.
        right_input = make_neuron_column(5, INPUT_COLOR, radius=0.075, buff=0.22, fill_opacity=0.14)
        right_hidden_1 = make_neuron_column(5, BN_COLOR, radius=0.072, buff=0.16, fill_opacity=0.08)
        right_hidden_2 = make_neuron_column(4, BN_COLOR, radius=0.072, buff=0.20, fill_opacity=0.08)
        right_output = make_neuron_column(4, NEUTRAL_COLOR, radius=0.070, buff=0.22, fill_opacity=0.08)
        bn_block_1 = make_bn_block()
        bn_block_2 = make_bn_block()

        right_input.move_to(RIGHT * 0.90 + UP * 0.35)
        right_hidden_1.move_to(RIGHT * 2.05 + UP * 0.35)
        bn_block_1.move_to(RIGHT * 2.85 + UP * 0.35)
        right_hidden_2.move_to(RIGHT * 3.70 + UP * 0.35)
        bn_block_2.move_to(RIGHT * 4.45 + UP * 0.35)
        right_output.move_to(RIGHT * 5.20 + UP * 0.35)

        right_in_to_h1 = connect_layers(right_input, right_hidden_1, BN_COLOR, opacity=0.13, width=0.9)
        right_h1_to_h2 = connect_layers(right_hidden_1, right_hidden_2, BN_COLOR, opacity=0.12, width=0.9)
        right_h2_to_out = connect_layers(right_hidden_2, right_output, BN_COLOR, opacity=0.12, width=0.9)

        right_input_label = make_layer_caption("same frozen\nCLIP features", INPUT_COLOR)
        right_hidden_1_label = make_layer_caption("256 units", BN_COLOR)
        right_hidden_2_label = make_layer_caption("128 units", BN_COLOR)
        right_output_label = make_layer_caption("138 names", LABEL_COLOR)
        bn_note = make_layer_caption("BatchNorm re-centers\nand re-scales", BN_COLOR)

        right_input_label.next_to(right_input, UP, buff=0.18)
        right_hidden_1_label.next_to(right_hidden_1, DOWN, buff=0.14)
        right_hidden_2_label.next_to(right_hidden_2, DOWN, buff=0.14)
        right_output_label.next_to(right_output, DOWN, buff=0.14)
        bn_note.move_to(RIGHT * 3.65 + DOWN * 1.65)

        left_result = make_result_card("930K params", "~9.8x", WIDE_COLOR)
        right_result = make_result_card("248K params", "~10.3x", BN_COLOR)
        left_result.move_to(LEFT * 3.30 + DOWN * 2.88)
        right_result.move_to(RIGHT * 3.15 + DOWN * 2.88)

        footer = caption_text(
            "Normalization matters more than brute-force width",
            scale=0.40,
            color=WHITE,
        )
        footer.to_edge(DOWN, buff=0.10)

        emphasis = SurroundingRectangle(right_result, buff=0.12, stroke_color=BN_COLOR, stroke_width=2.2)

        def play_bn_pass(block, companion_animation=None, run_time=1.35):
            traveling = block.left_bars.copy()
            targets = VGroup(*[bar.copy() for bar in block.right_bars])
            for mob in traveling:
                mob.set_fill(BN_COLOR, opacity=0.95)
                mob.set_stroke(BN_COLOR, width=0.0, opacity=0.0)
            self.add(traveling)

            animations = [
                ShowPassingFlash(
                    block.box.copy().set_stroke(BN_COLOR, width=4.0, opacity=0.95),
                    time_width=0.75,
                ),
                ShowPassingFlash(
                    block.inner_arrow.copy().set_stroke(BN_COLOR, width=3.0, opacity=0.95),
                    time_width=0.75,
                ),
                block.left_bars.animate.set_fill(BN_COLOR, opacity=0.35),
                block.right_bars.animate.set_fill(BN_COLOR, opacity=0.95),
                LaggedStart(
                    *[
                        Transform(source_bar, target_bar)
                        for source_bar, target_bar in zip(traveling, targets)
                    ],
                    lag_ratio=0.18,
                ),
            ]
            if companion_animation is not None:
                animations.append(companion_animation)

            self.play(*animations, run_time=run_time)
            self.remove(traveling)

        self.play(FadeIn(title, shift=DOWN * 0.10), run_time=0.6)
        self.play(
            FadeIn(left_header, shift=UP * 0.08),
            FadeIn(right_header, shift=UP * 0.08),
            run_time=0.7,
        )
        self.play(
            FadeIn(left_input, scale=0.7),
            FadeIn(left_hidden, scale=0.7),
            FadeIn(left_output, scale=0.7),
            ShowCreation(left_in_to_hidden),
            ShowCreation(left_hidden_to_out),
            FadeIn(left_input_label, shift=UP * 0.05),
            FadeIn(left_hidden_label, shift=UP * 0.05),
            FadeIn(left_output_label, shift=UP * 0.05),
            run_time=0.9,
        )
        self.play(
            FadeIn(right_input, scale=0.7),
            FadeIn(right_hidden_1, scale=0.7),
            FadeIn(right_hidden_2, scale=0.7),
            FadeIn(right_output, scale=0.7),
            ShowCreation(right_in_to_h1),
            ShowCreation(right_h1_to_h2),
            ShowCreation(right_h2_to_out),
            FadeIn(bn_block_1, scale=0.92),
            FadeIn(bn_block_2, scale=0.92),
            FadeIn(right_input_label, shift=UP * 0.05),
            FadeIn(right_hidden_1_label, shift=UP * 0.05),
            FadeIn(right_hidden_2_label, shift=UP * 0.05),
            FadeIn(right_output_label, shift=UP * 0.05),
            run_time=1.0,
        )

        self.play(
            pulse_lines(left_in_to_hidden, INPUT_COLOR),
            pulse_lines(right_in_to_h1, INPUT_COLOR),
            run_time=0.9,
        )

        left_scales = [1.30, 0.62, 1.18, 0.54, 1.42, 0.48, 1.10, 0.58, 1.25]
        left_opacities = [0.92, 0.22, 0.84, 0.18, 0.95, 0.16, 0.78, 0.24, 0.88]
        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(
                        dot.animate.scale(scale).set_fill(WIDE_COLOR, opacity=opacity).set_stroke(WIDE_COLOR, width=1.4, opacity=max(0.35, opacity)),
                        lag_ratio=0.0,
                    )
                    for dot, scale, opacity in zip(left_hidden, left_scales, left_opacities)
                ],
                lag_ratio=0.03,
            ),
            FadeIn(left_takeaway, shift=UP * 0.06),
            run_time=0.9,
        )

        right_raw_scales = [1.26, 0.64, 1.18, 0.70, 1.12]
        right_raw_opacities = [0.80, 0.26, 0.76, 0.30, 0.72]
        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(
                        dot.animate.scale(scale).set_fill(BN_COLOR, opacity=opacity).set_stroke(BN_COLOR, width=1.35, opacity=max(0.35, opacity)),
                        lag_ratio=0.0,
                    )
                    for dot, scale, opacity in zip(right_hidden_1, right_raw_scales, right_raw_opacities)
                ],
                lag_ratio=0.03,
            ),
            run_time=0.9,
        )

        hidden2_balanced_scales = [1.02, 1.00, 1.03, 1.01]
        play_bn_pass(
            bn_block_1,
            companion_animation=AnimationGroup(
                pulse_lines(right_h1_to_h2, BN_COLOR),
                AnimationGroup(
                    *[
                        AnimationGroup(
                            dot.animate.scale(scale).set_fill(BN_COLOR, opacity=0.85).set_stroke(BN_COLOR, width=1.35, opacity=0.85),
                            lag_ratio=0.0,
                        )
                        for dot, scale in zip(right_hidden_2, hidden2_balanced_scales)
                    ],
                    lag_ratio=0.05,
                ),
                lag_ratio=0.0,
            ),
            run_time=1.55,
        )

        output_scales = [1.03, 1.01, 1.02, 1.00]
        play_bn_pass(
            bn_block_2,
            companion_animation=AnimationGroup(
                pulse_lines(right_h2_to_out, BN_COLOR),
                AnimationGroup(
                    *[
                        AnimationGroup(
                            dot.animate.scale(scale).set_fill(NEUTRAL_COLOR, opacity=0.18).set_stroke(WHITE, width=1.4, opacity=0.85),
                            lag_ratio=0.0,
                        )
                        for dot, scale in zip(right_output, output_scales)
                    ],
                    lag_ratio=0.05,
                ),
                FadeIn(bn_note, shift=UP * 0.06),
                lag_ratio=0.0,
            ),
            run_time=1.55,
        )

        self.play(
            AnimationGroup(
                *[
                    AnimationGroup(
                        dot.animate.scale(1 / scale),
                        lag_ratio=0.0,
                    )
                    for dot, scale in zip(right_output, output_scales)
                ],
                lag_ratio=0.04,
            ),
            run_time=0.3,
        )

        self.play(
            FadeIn(left_result, shift=UP * 0.08),
            FadeIn(right_result, shift=UP * 0.08),
            run_time=0.65,
        )
        self.play(
            ShowCreation(emphasis),
            FadeIn(footer, shift=UP * 0.10),
            run_time=0.65,
        )

        verifier.check_inside_frame("title", title, margin=0.08)
        verifier.check_inside_frame("left_header", left_header, margin=0.08)
        verifier.check_inside_frame("right_header", right_header, margin=0.08)
        verifier.check_inside_frame("left_input", left_input, margin=0.08)
        verifier.check_inside_frame("left_hidden", left_hidden, margin=0.08)
        verifier.check_inside_frame("left_output", left_output, margin=0.08)
        verifier.check_inside_frame("right_input", right_input, margin=0.08)
        verifier.check_inside_frame("right_hidden_1", right_hidden_1, margin=0.08)
        verifier.check_inside_frame("right_hidden_2", right_hidden_2, margin=0.08)
        verifier.check_inside_frame("right_output", right_output, margin=0.08)
        verifier.check_inside_frame("left_result", left_result, margin=0.08)
        verifier.check_inside_frame("right_result", right_result, margin=0.08)
        verifier.check_inside_frame("footer", footer, margin=0.08)
        verifier.check_vertical_order("title", title, "left_header", left_header, min_gap=0.12)
        verifier.check_vertical_order("title", title, "right_header", right_header, min_gap=0.12)
        verifier.check_vertical_order("left_output", left_output, "left_result", left_result, min_gap=0.18)
        verifier.check_vertical_order("right_output", right_output, "right_result", right_result, min_gap=0.18)
        verifier.check_vertical_order("left_result", left_result, "footer", footer, min_gap=0.12)
        verifier.check_vertical_order("right_result", right_result, "footer", footer, min_gap=0.12)
        verifier.check_min_horizontal_gap("left_output", left_output, "right_input", right_input, min_gap=0.65)
        verifier.assert_ok()

        self.wait(1.5)
