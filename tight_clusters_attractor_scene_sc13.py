from manimlib import *
import numpy as np
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distribution_metrics_shared import BG_COLOR, LABEL_COLOR, caption_text
from layout_verifier import LayoutVerifier


PANEL_STROKE = "#5c647d"
DIFFUSE_COLOR = BLUE_D
TIGHT_COLOR = ORANGE
QUERY_COLOR = GREY_B
QUERY_SELECTED_GLOW = YELLOW


def cluster_points(n, center, spread_x, spread_y, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(center[0], spread_x, size=n)
    y = rng.normal(center[1], spread_y, size=n)
    return np.column_stack([x, y])


def make_score_label(text, color):
    return Text(text, font="Helvetica Neue", color=color, font_size=18)


class TightClustersAttractorScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        verifier = LayoutVerifier(scene_name="TightClustersAttractorScene")

        title = Text(
            "Tight Clusters Become Attractors",
            font="Helvetica Neue",
            color=WHITE,
            font_size=32,
        )
        title.to_edge(UP, buff=0.28)

        axes = Axes(
            x_range=[-3.8, 3.8, 1],
            y_range=[-2.5, 2.5, 1],
            width=8.6,
            height=4.8,
            axis_config=dict(
                include_ticks=False,
                include_tip=False,
                stroke_color=PANEL_STROKE,
                stroke_width=1.4,
            ),
        )
        frame = SurroundingRectangle(
            axes,
            buff=0.0,
            stroke_color=PANEL_STROKE,
            stroke_width=1.5,
        )
        plot_group = VGroup(axes, frame)
        plot_group.move_to(UP * 0.65)

        diffuse_data = cluster_points(34, center=(-1.2, -0.15), spread_x=0.95, spread_y=0.62, seed=4)
        tight_data = cluster_points(28, center=(1.35, 0.4), spread_x=0.28, spread_y=0.24, seed=7)

        diffuse_dots = VGroup(*[
            Dot(
                axes.c2p(x, y),
                radius=0.056,
                color=DIFFUSE_COLOR,
                fill_opacity=0.98,
                stroke_color=DIFFUSE_COLOR,
                stroke_width=0.6,
                stroke_opacity=0.95,
            )
            for x, y in diffuse_data
        ])
        tight_dots = VGroup(*[
            Dot(
                axes.c2p(x, y),
                radius=0.056,
                color=TIGHT_COLOR,
                fill_opacity=0.98,
                stroke_color=TIGHT_COLOR,
                stroke_width=0.6,
                stroke_opacity=0.95,
            )
            for x, y in tight_data
        ])

        diffuse_center_xy = diffuse_data.mean(axis=0)
        tight_center_xy = tight_data.mean(axis=0)
        diffuse_center = Dot(axes.c2p(*diffuse_center_xy), radius=0.07, color=DIFFUSE_COLOR)
        tight_center = Dot(axes.c2p(*tight_center_xy), radius=0.07, color=TIGHT_COLOR)
        diffuse_center.set_stroke(WHITE, width=1.0, opacity=0.55)
        tight_center.set_stroke(WHITE, width=1.0, opacity=0.55)

        diffuse_label = caption_text("larger, diffuse cluster", scale=0.34, color=DIFFUSE_COLOR)
        tight_label = caption_text("small, tight cluster", scale=0.34, color=TIGHT_COLOR)
        diffuse_label.next_to(axes.c2p(-1.75, 1.55), UP, buff=0.08)
        tight_label.next_to(axes.c2p(1.35, 1.65), UP, buff=0.08)

        diffuse_name = caption_text("David-like region", scale=0.31, color=DIFFUSE_COLOR)
        tight_name = caption_text("Todd-like region", scale=0.31, color=TIGHT_COLOR)
        diffuse_name.next_to(diffuse_label, UP, buff=0.06)
        tight_name.next_to(tight_label, UP, buff=0.06)

        query_coords = [
            (-0.20, 0.22),
            (0.02, -0.18),
            (0.18, 0.46),
        ]
        query_dots = VGroup(*[
            Dot(
                axes.c2p(x, y),
                radius=0.062,
                color=QUERY_COLOR,
                fill_opacity=0.45,
                stroke_color=WHITE,
                stroke_width=1.2,
                stroke_opacity=0.8,
            )
            for x, y in query_coords
        ])
        query_glows = VGroup(*[
            Dot(
                dot.get_center(),
                radius=0.12,
                color=QUERY_SELECTED_GLOW,
                fill_opacity=0.0,
                stroke_color=QUERY_SELECTED_GLOW,
                stroke_width=2.5,
                stroke_opacity=0.0,
            )
            for dot in query_dots
        ])

        query_label = caption_text("ambiguous query points", scale=0.34, color=WHITE)
        query_label.next_to(query_dots, DOWN, buff=0.28).shift(LEFT * 0.12)

        diffuse_lines = VGroup()
        tight_lines = VGroup()
        for i, dot in enumerate(query_dots):
            start = dot.get_center()
            line_diffuse = Line(
                start,
                diffuse_center.get_center(),
                stroke_color=DIFFUSE_COLOR,
                stroke_width=2.2,
                stroke_opacity=0.42,
            )
            line_tight = Line(
                start,
                tight_center.get_center(),
                stroke_color=TIGHT_COLOR,
                stroke_width=3.6,
                stroke_opacity=0.82,
            )
            diffuse_lines.add(line_diffuse)
            tight_lines.add(line_tight)
        diffuse_footprint = DashedVMobject(
            Ellipse(
                width=4.7,
                height=3.0,
                stroke_color=DIFFUSE_COLOR,
                stroke_width=2.0,
            ),
            num_dashes=34,
        )
        diffuse_footprint.move_to(axes.c2p(-1.15, -0.10))

        tight_footprint = DashedVMobject(
            Ellipse(
                width=1.65,
                height=1.20,
                stroke_color=TIGHT_COLOR,
                stroke_width=2.3,
            ),
            num_dashes=22,
        )
        tight_footprint.move_to(axes.c2p(1.28, 0.35))

        tight_region = Ellipse(
            width=4.6,
            height=3.0,
            stroke_color=TIGHT_COLOR,
            stroke_width=2.8,
            fill_color=TIGHT_COLOR,
            fill_opacity=0.15,
        )
        tight_region.move_to(axes.c2p(1.15, 0.20))

        footprint_note = caption_text(
            "small dashed outline = actual tight cluster",
            scale=0.31,
            color=TIGHT_COLOR,
        )
        footprint_note.move_to(axes.c2p(1.55, 2.12))

        territory_note = caption_text(
            "large yellow region = predicted as Todd",
            scale=0.31,
            color=TIGHT_COLOR,
        )
        territory_note.move_to(axes.c2p(1.78, -1.85))

        region_label = caption_text(
            "A small cluster ends up owning a much larger prediction region",
            scale=0.38,
            color=WHITE,
        )
        region_label.move_to(DOWN * 2.45)

        routing_note = caption_text(
            "Uncertain points keep getting routed to Todd",
            scale=0.36,
            color=WHITE,
        )
        routing_note.move_to(DOWN * 2.55)

        outcome_note = caption_text(
            "Tight clusters can attract uncertain predictions",
            scale=0.42,
            color=WHITE,
        )
        outcome_note.move_to(DOWN * 3.10)

        oversized_arrow = Arrow(
            region_label.get_top() + UP * 0.05,
            tight_region.get_bottom() + DOWN * 0.10,
            buff=0.08,
            stroke_color=TIGHT_COLOR,
            stroke_width=2.8,
        )

        self.play(FadeIn(title, shift=DOWN * 0.10), run_time=0.65)
        self.play(ShowCreation(axes), ShowCreation(frame), run_time=0.75)

        self.play(
            LaggedStart(*[FadeIn(dot, scale=0.5) for dot in diffuse_dots], lag_ratio=0.03),
            FadeIn(diffuse_label, shift=UP * 0.08),
            FadeIn(diffuse_name, shift=UP * 0.08),
            FadeIn(diffuse_center, scale=0.7),
            run_time=1.05,
        )
        self.play(
            LaggedStart(*[FadeIn(dot, scale=0.5) for dot in tight_dots], lag_ratio=0.03),
            FadeIn(tight_label, shift=UP * 0.08),
            FadeIn(tight_name, shift=UP * 0.08),
            FadeIn(tight_center, scale=0.7),
            run_time=1.0,
        )

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        FadeIn(dot, scale=0.6),
                        FadeIn(glow, scale=0.6),
                    )
                    for dot, glow in zip(query_dots, query_glows)
                ],
                lag_ratio=0.15,
            ),
            FadeIn(query_label, shift=UP * 0.08),
            run_time=0.9,
        )

        self.play(ShowCreation(diffuse_footprint), run_time=0.55)
        self.play(ShowCreation(tight_footprint), FadeIn(footprint_note, shift=UP * 0.06), run_time=0.6)

        for dot, glow, diffuse_line, tight_line in zip(
            query_dots, query_glows, diffuse_lines, tight_lines
        ):
            self.play(
                ShowCreation(diffuse_line),
                ShowCreation(tight_line),
                run_time=0.55,
            )
            self.play(
                AnimationGroup(
                    dot.animate.set_fill(WHITE, opacity=0.92).set_stroke(QUERY_SELECTED_GLOW, width=1.8, opacity=0.95).scale(1.06),
                    glow.animate.set_fill(QUERY_SELECTED_GLOW, opacity=0.22).set_stroke(QUERY_SELECTED_GLOW, opacity=0.9).scale(1.10),
                    lag_ratio=0.0,
                ),
                run_time=0.34,
            )
            self.play(
                AnimationGroup(
                    dot.animate.scale(1 / 1.06),
                    glow.animate.scale(1 / 1.10),
                    lag_ratio=0.0,
                ),
                run_time=0.16,
            )

        self.play(FadeIn(routing_note, shift=UP * 0.08), run_time=0.55)
        self.play(
            FadeIn(tight_region),
            FadeIn(territory_note, shift=UP * 0.06),
            run_time=0.75,
        )
        self.play(
            FadeOut(routing_note, shift=DOWN * 0.06),
            GrowArrow(oversized_arrow),
            FadeIn(region_label, shift=UP * 0.10),
            FadeIn(outcome_note, shift=UP * 0.10),
            run_time=0.75,
        )

        verifier.check_inside_frame("title", title, margin=0.08)
        verifier.check_inside_frame("plot_group", plot_group, margin=0.08)
        verifier.check_inside_frame("diffuse_label", diffuse_label, margin=0.08)
        verifier.check_inside_frame("tight_label", tight_label, margin=0.08)
        verifier.check_inside_frame("query_label", query_label, margin=0.08)
        verifier.check_inside_frame("footprint_note", footprint_note, margin=0.08)
        verifier.check_inside_frame("territory_note", territory_note, margin=0.08)
        verifier.check_inside_frame("region_label", region_label, margin=0.08)
        verifier.check_inside_frame("outcome_note", outcome_note, margin=0.08)
        verifier.check_vertical_order("title", title, "plot_group", plot_group, min_gap=0.12)
        verifier.check_vertical_order("plot_group", plot_group, "region_label", region_label, min_gap=0.16)
        verifier.check_vertical_order("region_label", region_label, "outcome_note", outcome_note, min_gap=0.16)
        verifier.check_min_horizontal_gap("diffuse_label", diffuse_label, "tight_label", tight_label, min_gap=0.50)
        verifier.check_no_overlap("query_label", query_label, "region_label", region_label, min_gap=0.06)
        verifier.check_no_overlap("footprint_note", footprint_note, "tight_region", tight_region, min_gap=0.04)
        verifier.check_no_overlap("territory_note", territory_note, "tight_footprint", tight_footprint, min_gap=0.04)
        verifier.check_no_overlap("tight_name", tight_name, "tight_region", tight_region, min_gap=0.02)
        verifier.check_no_overlap("diffuse_name", diffuse_name, "diffuse_footprint", diffuse_footprint, min_gap=0.02)
        verifier.assert_ok()

        self.wait(1.5)
