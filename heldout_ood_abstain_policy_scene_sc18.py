from manimlib import *
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distribution_metrics_shared import BG_COLOR, LABEL_COLOR, caption_text
from layout_verifier import LayoutVerifier


PANEL_STROKE = "#5c647d"
IN_VOCAB_COLOR = GREEN_C
HELDOUT_COLOR = YELLOW_D
LIVE_COLOR = RED_C
PROBE_COLOR = BLUE_C
CENTROID_COLOR = TEAL_C
ABSTAIN_COLOR = ORANGE


def make_bucket_card(title, subtitle, color, bar_fill):
    box = RoundedRectangle(
        width=2.30,
        height=1.20,
        corner_radius=0.12,
        stroke_color=color,
        stroke_width=1.8,
        fill_color=color,
        fill_opacity=0.06,
    )
    title_mob = caption_text(title, scale=0.34, color=color)
    subtitle_mob = caption_text(subtitle, scale=0.25, color=LABEL_COLOR)
    rail = RoundedRectangle(
        width=1.55,
        height=0.12,
        corner_radius=0.05,
        stroke_width=0,
        fill_color=GREY_E,
        fill_opacity=0.55,
    )
    fill = RoundedRectangle(
        width=1.55 * bar_fill,
        height=0.12,
        corner_radius=0.05,
        stroke_width=0,
        fill_color=color,
        fill_opacity=0.95,
    )
    fill.align_to(rail, LEFT)
    bar = VGroup(rail, fill)
    content = VGroup(title_mob, subtitle_mob, bar).arrange(DOWN, buff=0.12)
    content.move_to(box.get_center())
    return VGroup(box, content)


def make_prediction_pill(label, color, width=1.05):
    pill = RoundedRectangle(
        width=width,
        height=0.36,
        corner_radius=0.16,
        stroke_color=color,
        stroke_width=1.3,
        fill_color=color,
        fill_opacity=0.14,
    )
    text = Text(label, font="Helvetica Neue", color=color, font_size=18)
    text.move_to(pill.get_center())
    return VGroup(pill, text)


def make_example_row(sample_label, probe_name, centroid_name, disagree=False):
    sample = RoundedRectangle(
        width=0.62,
        height=0.44,
        corner_radius=0.08,
        stroke_color=WHITE,
        stroke_width=1.1,
        fill_color=GREY_B,
        fill_opacity=0.18,
    )
    sample_text = Text(sample_label, font="Helvetica Neue", color=WHITE, font_size=16)
    sample_text.move_to(sample.get_center())
    sample_group = VGroup(sample, sample_text)

    probe_tag = caption_text("probe", scale=0.24, color=PROBE_COLOR)
    probe_pill = make_prediction_pill(probe_name, PROBE_COLOR)
    probe_group = VGroup(probe_tag, probe_pill).arrange(DOWN, buff=0.06)

    centroid_tag = caption_text("centroid", scale=0.24, color=CENTROID_COLOR)
    centroid_pill = make_prediction_pill(centroid_name, CENTROID_COLOR, width=1.16)
    centroid_group = VGroup(centroid_tag, centroid_pill).arrange(DOWN, buff=0.06)

    row = VGroup(sample_group, probe_group, centroid_group).arrange(RIGHT, buff=0.24, aligned_edge=DOWN)
    disagree_badge = None
    if disagree:
        disagree_badge = make_prediction_pill("disagree", RED_C, width=0.95)
        disagree_badge.scale(0.78)
        disagree_badge.next_to(row, RIGHT, buff=0.16)
        row = VGroup(row, disagree_badge)

    return row


def make_policy_gate():
    diamond = Square(side_length=1.12)
    diamond.rotate(PI / 4)
    diamond.set_stroke(ABSTAIN_COLOR, width=2.0)
    diamond.set_fill(ABSTAIN_COLOR, opacity=0.10)
    text = Text("abstain\npolicy", font="Helvetica Neue", color=ABSTAIN_COLOR, font_size=18)
    text.move_to(diamond.get_center())
    gate = VGroup(diamond, text)

    rule_1 = caption_text("low confidence", scale=0.25, color=ABSTAIN_COLOR)
    rule_2 = caption_text("or disagreement", scale=0.25, color=ABSTAIN_COLOR)
    rules = VGroup(rule_1, rule_2).arrange(DOWN, buff=0.05)
    rules.next_to(gate, DOWN, buff=0.16)
    return VGroup(gate, rules)


def make_outcome_card(title, line_1, line_2, color):
    box = RoundedRectangle(
        width=2.55,
        height=1.18,
        corner_radius=0.14,
        stroke_color=color,
        stroke_width=1.8,
        fill_color=color,
        fill_opacity=0.08,
    )
    top = caption_text(title, scale=0.31, color=color)
    mid = caption_text(line_1, scale=0.28, color=WHITE)
    bottom = caption_text(line_2, scale=0.26, color=LABEL_COLOR)
    text_group = VGroup(top, mid, bottom).arrange(DOWN, buff=0.08)
    text_group.move_to(box.get_center())
    return VGroup(box, text_group)


class HeldOutOODAbstainPolicyScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        verifier = LayoutVerifier(scene_name="HeldOutOODAbstainPolicyScene")

        title = Text(
            "Held-Out OOD Evaluation and Abstain Policy",
            font="Helvetica Neue",
            color=WHITE,
            font_size=31,
        )
        title.to_edge(UP, buff=0.28)

        bucket_1 = make_bucket_card("in-vocab validation", "known names", IN_VOCAB_COLOR, 0.86)
        bucket_2 = make_bucket_card("held-out names", "never trained", HELDOUT_COLOR, 0.48)
        bucket_3 = make_bucket_card("live uploads", "messy real inputs", LIVE_COLOR, 0.34)
        buckets = VGroup(bucket_1, bucket_2, bucket_3).arrange(RIGHT, buff=0.28)
        buckets.move_to(UP * 1.92)

        bucket_note = caption_text(
            "Confidence and reliability drop as inputs move away from training conditions",
            scale=0.30,
            color=WHITE,
        )
        bucket_note.next_to(buckets, DOWN, buff=0.14)

        row_1 = make_example_row("A", "David", "David", disagree=False)
        row_2 = make_example_row("B", "Todd", "Lucas", disagree=True)
        row_3 = make_example_row("C", "Pierre", "Emily", disagree=True)
        example_rows = VGroup(row_1, row_2, row_3).arrange(DOWN, buff=0.20, aligned_edge=LEFT)
        example_rows.move_to(LEFT * 2.45 + DOWN * 0.50)

        example_label = caption_text("uncertain examples can split probe vs centroid", scale=0.31, color=WHITE)
        example_label.next_to(example_rows, UP, buff=0.14)

        gate = make_policy_gate()
        gate.move_to(RIGHT * 2.95 + DOWN * 0.35)

        arrows_to_gate = VGroup(*[
            Arrow(
                row.get_right() + RIGHT * 0.06,
                gate[0].get_left() + LEFT * 0.06 + DOWN * offset,
                buff=0.04,
                stroke_color=ABSTAIN_COLOR,
                stroke_width=2.2,
            )
            for row, offset in zip(example_rows, [0.18, 0.0, -0.18])
        ])

        before_card = make_outcome_card(
            "before: always answer",
            "more confident wrong guesses",
            "coverage stays high",
            RED_C,
        )
        after_card = make_outcome_card(
            "after: abstain when unsure",
            "fewer bad public predictions",
            "coverage gets lower",
            GREEN_C,
        )
        before_card.move_to(LEFT * 3.15 + DOWN * 2.90)
        after_card.move_to(RIGHT * 3.15 + DOWN * 2.90)

        before_arrow = Arrow(
            example_rows.get_bottom() + DOWN * 0.08,
            before_card.get_top() + UP * 0.05,
            buff=0.04,
            stroke_color=RED_C,
            stroke_width=2.4,
        )
        after_arrow = Arrow(
            gate.get_bottom() + DOWN * 0.08,
            after_card.get_top() + UP * 0.05,
            buff=0.04,
            stroke_color=GREEN_C,
            stroke_width=2.4,
        )

        abstain_chip = make_prediction_pill("not sure", ABSTAIN_COLOR, width=1.18)
        abstain_chip.move_to(after_card.get_top() + UP * 0.68)

        footer = caption_text(
            "For a public demo, abstaining can be better than guessing",
            scale=0.40,
            color=WHITE,
        )
        footer.to_edge(DOWN, buff=0.08)

        self.play(FadeIn(title, shift=DOWN * 0.10), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(bucket, shift=UP * 0.08) for bucket in buckets], lag_ratio=0.12),
            run_time=0.95,
        )
        self.play(FadeIn(bucket_note, shift=UP * 0.06), run_time=0.45)
        self.play(
            FadeIn(example_label, shift=UP * 0.06),
            LaggedStart(*[FadeIn(row, shift=RIGHT * 0.06) for row in example_rows], lag_ratio=0.14),
            run_time=0.95,
        )
        self.play(
            FadeIn(gate, scale=0.92),
            LaggedStart(*[GrowArrow(arrow) for arrow in arrows_to_gate], lag_ratio=0.10),
            run_time=0.9,
        )
        self.play(
            GrowArrow(before_arrow),
            GrowArrow(after_arrow),
            FadeIn(before_card, shift=UP * 0.08),
            FadeIn(after_card, shift=UP * 0.08),
            run_time=0.8,
        )
        self.play(
            FadeIn(abstain_chip, scale=0.88),
            FadeIn(footer, shift=UP * 0.10),
            run_time=0.65,
        )

        verifier.check_inside_frame("title", title, margin=0.08)
        verifier.check_inside_frame("buckets", buckets, margin=0.08)
        verifier.check_inside_frame("bucket_note", bucket_note, margin=0.08)
        verifier.check_inside_frame("example_rows", example_rows, margin=0.08)
        verifier.check_inside_frame("gate", gate, margin=0.08)
        verifier.check_inside_frame("before_card", before_card, margin=0.08)
        verifier.check_inside_frame("after_card", after_card, margin=0.08)
        verifier.check_inside_frame("footer", footer, margin=0.08)
        verifier.check_vertical_order("title", title, "buckets", buckets, min_gap=0.12)
        verifier.check_vertical_order("buckets", buckets, "bucket_note", bucket_note, min_gap=0.10)
        verifier.check_vertical_order("bucket_note", bucket_note, "example_label", example_label, min_gap=0.18)
        verifier.check_vertical_order("example_rows", example_rows, "before_card", before_card, min_gap=0.16)
        verifier.check_vertical_order("gate", gate, "after_card", after_card, min_gap=0.16)
        verifier.check_vertical_order("before_card", before_card, "footer", footer, min_gap=0.12)
        verifier.check_vertical_order("after_card", after_card, "footer", footer, min_gap=0.12)
        verifier.check_min_horizontal_gap("example_rows", example_rows, "gate", gate, min_gap=0.45)
        verifier.check_min_horizontal_gap("before_card", before_card, "after_card", after_card, min_gap=0.65)
        verifier.assert_ok()

        self.wait(1.5)
