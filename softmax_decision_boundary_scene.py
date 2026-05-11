from manimlib import *
import numpy as np


class SoftmaxDecisionBoundaryScene(Scene):
    def make_face_crop(self, side=1.35):
        image = ImageMobject("croped1.png")
        image.set_height(side)
        if image.get_width() > side:
            image.set_width(side)

        frame = Square(side_length=side + 0.05)
        frame.set_fill("#111318", opacity=0.18)
        frame.set_stroke(WHITE, width=1.5, opacity=0.88)
        frame.move_to(image)

        label = TexText("face crop", font_size=18, color=GREY_A)
        label.next_to(frame, DOWN, buff=0.12)
        return Group(image, frame, label)

    def make_arrow_between(self, left, right, color=GREY_A, buff=0.14):
        return Arrow(
            left.get_right() + RIGHT * buff,
            right.get_left() - RIGHT * buff,
            buff=0,
            fill_color=color,
            stroke_width=2.0,
            thickness=0.025,
            max_tip_length_to_length_ratio=0.13,
        )

    def make_mlp_network(self):
        layer_sizes = [4, 5, 4]
        layer_colors = [YELLOW_B, GREEN_B, BLUE_B]
        node_layers = VGroup()
        for layer_index, size in enumerate(layer_sizes):
            nodes = VGroup()
            for _ in range(size):
                node = Circle(radius=0.085)
                node.set_fill(layer_colors[layer_index], opacity=0.10)
                node.set_stroke(layer_colors[layer_index], width=1.35, opacity=0.92)
                nodes.add(node)
            nodes.arrange(DOWN, buff=0.17)
            node_layers.add(nodes)
        node_layers.arrange(RIGHT, buff=0.72)

        connections = VGroup()
        for left_layer, right_layer in zip(node_layers[:-1], node_layers[1:]):
            for left_node in left_layer:
                for right_node in right_layer:
                    line = Line(left_node.get_center(), right_node.get_center())
                    line.set_stroke(GREY_B, width=0.65, opacity=0.28)
                    connections.add(line)

        label = TexText("MLP classifier", font_size=20, color=WHITE)
        label.next_to(node_layers, DOWN, buff=0.18)
        return VGroup(connections, node_layers, label)

    def make_bar_rows(self, title_text, names, values, colors, max_width=2.2, value_fmt="{:.2f}", title_color=WHITE):
        rows = VGroup()
        max_value = max(values)
        label_right_x = -0.38
        bar_left_x = -0.18
        row_step = 0.45
        for index, (name, value, color) in enumerate(zip(names, values, colors)):
            y = -index * row_step
            row = VGroup()
            label = TexText(name, font_size=17, color=WHITE)
            bar = Rectangle(width=max_width * value / max_value, height=0.28)
            bar.set_fill(color, opacity=0.62)
            bar.set_stroke(color, width=1.15, opacity=0.96)
            value_label = Tex(value_fmt.format(value), font_size=17, color=color)
            label.move_to([label_right_x - label.get_width() / 2, y, 0])
            bar.move_to([bar_left_x + bar.get_width() / 2, y, 0])
            value_label.next_to(bar, RIGHT, buff=0.16)
            row.add(label, bar, value_label)
            rows.add(row)

        title = TexText(title_text, font_size=22, color=title_color)
        title.next_to(rows, UP, buff=0.18)
        return VGroup(title, rows)

    def make_probability_mass_bar(self):
        names = ["Anna", "Daniel", "Mary", "Chris", "Other"]
        values = [0.31, 0.27, 0.22, 0.13, 0.07]
        colors = [YELLOW_B, BLUE_B, TEAL_B, GREEN_B, GREY_B]
        total_width = 4.6
        pieces = VGroup()
        x = -total_width / 2
        for value, color in zip(values, colors):
            piece = Rectangle(width=total_width * value, height=0.42)
            piece.set_fill(color, opacity=0.70)
            piece.set_stroke(BLACK, width=0.6, opacity=0.7)
            piece.move_to([x + piece.get_width() / 2, 0, 0])
            pieces.add(piece)
            x += piece.get_width()
        label = Tex(r"\sum_i p_i = 1", font_size=31, color=WHITE)
        label.next_to(pieces, DOWN, buff=0.22)
        caption = TexText("fixed probability mass", font_size=22, color=GREY_A)
        caption.next_to(pieces, UP, buff=0.18)
        return VGroup(caption, pieces, label)

    def make_close_race(self):
        names = ["Anna", "Daniel"]
        logit_values = [2.10, 2.00]
        prob_values = [0.525, 0.475]
        colors = [YELLOW_B, BLUE_B]
        logits = self.make_bar_rows("close logits", names, logit_values, colors, max_width=2.25)
        probs = self.make_bar_rows("softmax probabilities", names, prob_values, colors, max_width=2.25, value_fmt="{:.3f}", title_color=YELLOW_B)
        winner = TexText("winner still chosen", font_size=20, color=YELLOW_B)
        winner.next_to(probs, DOWN, buff=0.20)
        return VGroup(logits, probs, winner)

    def make_decision_boundary(self):
        plane = RoundedRectangle(
            width=5.35,
            height=3.55,
            corner_radius=0.12,
            stroke_color=GREY_B,
            stroke_width=1.1,
            fill_color="#101217",
            fill_opacity=0.92,
        )

        boundary_top = plane.get_top() + RIGHT * 0.18
        boundary_bottom = plane.get_bottom() + LEFT * 0.12
        left_region = Polygon(
            plane.get_corner(UL),
            plane.get_corner(DL),
            boundary_bottom,
            boundary_top,
        )
        left_region.set_fill(BLUE_B, opacity=0.13)
        left_region.set_stroke(BLUE_B, width=1.0, opacity=0.30)

        attractor_region = Polygon(
            boundary_top,
            boundary_bottom,
            plane.get_corner(DR),
            plane.get_corner(UR),
        )
        attractor_region.set_fill(YELLOW_B, opacity=0.15)
        attractor_region.set_stroke(YELLOW_B, width=1.1, opacity=0.42)

        boundary_line = Line(boundary_top, boundary_bottom)
        boundary_line.set_stroke(WHITE, width=1.75, opacity=0.92)

        absorbed_region = Polygon(
            boundary_top,
            boundary_bottom,
            boundary_bottom + RIGHT * 0.26,
            boundary_top + RIGHT * 0.26,
        )
        absorbed_region.set_fill(YELLOW_B, opacity=0.0)
        absorbed_region.set_stroke(YELLOW_B, width=0.9, opacity=0.0)

        labels = VGroup(
            TexText("neighbor names", font_size=18, color=BLUE_B),
            TexText("attractor name", font_size=18, color=YELLOW_B),
            TexText("decision boundary", font_size=16, color=WHITE),
            TexText("borderline faces", font_size=15, color=WHITE),
        )
        labels[0].move_to(plane.get_left() + RIGHT * 1.25 + UP * 1.28)
        labels[1].move_to(plane.get_right() + LEFT * 1.18 + UP * 1.28)
        labels[2].move_to(plane.get_center() + DOWN * 1.28)
        labels[3].move_to(plane.get_center() + UP * 0.94)

        rng = np.random.default_rng(10)
        points_group = VGroup()
        for index in range(38):
            x = rng.normal(0.20, 0.58)
            y = rng.normal(-0.02, 0.58)
            x = np.clip(x, -1.55, 1.55)
            y = np.clip(y, -1.18, 1.18)
            dot = Dot(plane.get_center() + RIGHT * x + UP * y, radius=0.035)
            dot.set_color(BLUE_B if index % 3 else TEAL_B)
            dot.set_opacity(0.68)
            points_group.add(dot)

        borderline_points = VGroup()
        for index, point in enumerate([RIGHT * 0.12 + UP * 0.36, RIGHT * 0.08 + DOWN * 0.08, LEFT * 0.08 + DOWN * 0.55]):
            dot = Dot(plane.get_center() + point, radius=0.055)
            dot.set_color(BLUE_B if index % 2 == 0 else TEAL_B)
            dot.set_stroke(YELLOW_B, width=1.6, opacity=1.0)
            borderline_points.add(dot)

        title = TexText("decision boundary in CLIP space", font_size=23, color=WHITE)
        title.next_to(plane, UP, buff=0.18)
        return VGroup(
            title,
            plane,
            left_region,
            attractor_region,
            boundary_line,
            absorbed_region,
            labels,
            points_group,
            borderline_points,
        )

    def make_prediction_distribution(self, title_text, names, values, colors):
        chart_width = max(7.2, 0.48 * len(names) + 0.85)
        chart_height = 2.25
        baseline = Line(LEFT * chart_width / 2, RIGHT * chart_width / 2)
        baseline.set_stroke(GREY_B, width=1.0, opacity=0.7)
        bars = VGroup()
        labels = VGroup()
        max_value = max(values)
        bar_width = 0.28
        for index, (name, value, color) in enumerate(zip(names, values, colors)):
            bar = Rectangle(width=bar_width, height=chart_height * value / max_value)
            bar.set_fill(color, opacity=0.72)
            bar.set_stroke(color, width=0.9, opacity=0.96)
            x = -chart_width / 2 + 0.35 + index * 0.48
            bar.move_to([x, bar.get_height() / 2, 0])
            bars.add(bar)
            label = TexText(name, font_size=12, color=color)
            label.rotate(35 * DEGREES)
            label.next_to(bar, DOWN, buff=0.12)
            labels.add(label)

        title = TexText(title_text, font_size=24, color=WHITE)
        title.move_to(UP * (chart_height + 0.58))
        return VGroup(title, baseline, bars, labels)

    def construct(self):
        self.camera.background_color = BLACK

        subtitle = TexText(
            "validation set was balanced; this was not just name frequency",
            font_size=18,
            color=GREY_B,
        )
        subtitle.to_edge(DOWN, buff=0.28)

        frame = RoundedRectangle(
            width=11.7,
            height=6.15,
            corner_radius=0.14,
            stroke_color=GREEN_B,
            stroke_width=1.25,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        frame.move_to(DOWN * 0.04)
        title_anchor = frame.get_top() + DOWN * 0.27
        caption_anchor = frame.get_bottom() + UP * 0.34
        title = Tex(r"\text{Classifier final step}", font_size=36, color=WHITE)
        title.move_to(title_anchor)
        stage_caption = TexText("MLP outputs raw scores for every name", font_size=23, color=GREY_A)
        stage_caption.move_to(caption_anchor)

        face = self.make_face_crop(side=1.24)
        face[1].move_to(LEFT * 4.35)
        face[0].move_to(face[1])
        face[2].next_to(face[1], DOWN, buff=0.12)
        mlp = self.make_mlp_network()
        mlp.move_to(LEFT * 1.05)
        names = ["Anna", "Daniel", "Mary", "Chris", "Other"]
        logit_values = [2.10, 2.00, 1.82, 0.70, 0.18]
        colors = [YELLOW_B, BLUE_B, TEAL_B, GREEN_B, GREY_B]
        logits = self.make_bar_rows("logits", names, logit_values, colors, max_width=2.25)
        logits.move_to(RIGHT * 3.65)
        face_to_mlp = self.make_arrow_between(face[1], mlp, color=GREEN_B)
        mlp_to_logits = self.make_arrow_between(mlp, logits, color=GREEN_B)

        self.play(
            FadeIn(frame),
            FadeIn(title, shift=UP * 0.06),
            FadeIn(stage_caption, shift=UP * 0.05),
            FadeIn(subtitle, shift=UP * 0.04),
            FadeIn(face, shift=UP * 0.08),
            FadeIn(mlp, shift=UP * 0.08),
            FadeIn(logits, shift=UP * 0.08),
            run_time=0.95,
        )
        self.play(
            mlp[0].animate.set_stroke(GREEN_B, width=1.25, opacity=0.76),
            mlp[1][0].animate.set_fill(YELLOW_B, opacity=0.42),
            run_time=0.35,
        )
        self.play(
            mlp[1][0].animate.set_fill(YELLOW_B, opacity=0.10),
            mlp[1][1].animate.set_fill(GREEN_B, opacity=0.46),
            run_time=0.35,
        )
        self.play(
            mlp[1][1].animate.set_fill(GREEN_B, opacity=0.10),
            mlp[1][2].animate.set_fill(BLUE_B, opacity=0.46),
            ShowCreation(face_to_mlp),
            ShowCreation(mlp_to_logits),
            run_time=0.65,
        )
        self.wait(0.5)
        for index, row in enumerate(logits[1]):
            self.play(
                row[1].animate.set_fill(colors[index], opacity=0.92).set_stroke(colors[index], width=2.1, opacity=1.0),
                row[2].animate.set_color(WHITE),
                run_time=0.14,
            )
            self.play(
                row[1].animate.set_fill(colors[index], opacity=0.62).set_stroke(colors[index], width=1.15, opacity=0.96),
                row[2].animate.set_color(colors[index]),
                run_time=0.10,
            )
        self.wait(0.55)

        softmax_title = Tex(r"\text{Softmax: exponentiate, then normalize}", font_size=34, color=WHITE)
        softmax_title.move_to(title_anchor)
        softmax_caption = Tex(r"p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}", font_size=32, color=YELLOW_B)
        softmax_caption.move_to(caption_anchor + UP * 0.75)
        exp_values = [8.17, 7.39, 6.17, 2.01, 1.20]
        probs = [0.328, 0.297, 0.248, 0.081, 0.046]
        exp_bars = self.make_bar_rows("exponentiated scores", names, exp_values, colors, max_width=2.25)
        exp_bars.move_to(logits)
        prob_bars = self.make_bar_rows("probabilities", names, probs, colors, max_width=2.25, value_fmt="{:.3f}", title_color=YELLOW_B)
        prob_bars.move_to(logits)

        self.play(
            Transform(title, softmax_title),
            Transform(stage_caption, softmax_caption),
            Transform(logits, exp_bars),
            frame.animate.set_stroke(YELLOW_B, width=1.35, opacity=1.0),
            run_time=0.9,
        )
        self.wait(0.35)
        mass_bar = self.make_probability_mass_bar()
        mass_bar.move_to(LEFT * 1.05 + DOWN * 0.12)
        self.play(
            FadeOut(Group(face, face_to_mlp, mlp, mlp_to_logits), shift=LEFT * 0.12),
            Transform(logits, prob_bars),
            FadeIn(mass_bar, shift=UP * 0.08),
            run_time=0.9,
        )
        self.play(
            mass_bar[1][0].animate.set_fill(YELLOW_B, opacity=0.92),
            mass_bar[1][1].animate.set_fill(BLUE_B, opacity=0.48),
            run_time=0.45,
        )
        self.play(
            mass_bar[1][0].animate.set_fill(YELLOW_B, opacity=0.70),
            mass_bar[1][1].animate.set_fill(BLUE_B, opacity=0.70),
            run_time=0.35,
        )
        self.wait(0.5)

        close_title = Tex(r"\text{Small logit gaps can still choose a winner}", font_size=34, color=WHITE)
        close_title.move_to(title_anchor)
        close_caption = TexText("names compete for the same probability mass", font_size=23, color=GREY_A)
        close_caption.move_to(caption_anchor)
        close_race = self.make_close_race()
        close_race[0].move_to(LEFT * 2.85 + DOWN * 0.02)
        close_race[1].move_to(RIGHT * 2.15 + DOWN * 0.02)
        close_race[2].move_to(RIGHT * 2.15 + DOWN * 1.42)
        race_arrow = self.make_arrow_between(close_race[0], close_race[1], color=YELLOW_B, buff=0.22)

        self.play(
            FadeOut(VGroup(logits, mass_bar), shift=DOWN * 0.08),
            Transform(title, close_title),
            Transform(stage_caption, close_caption),
            FadeIn(close_race[0], shift=UP * 0.06),
            run_time=0.8,
        )
        self.play(
            ShowCreation(race_arrow),
            FadeIn(close_race[1], shift=RIGHT * 0.08),
            run_time=0.75,
        )
        self.play(
            close_race[1][1][0][1].animate.set_fill(YELLOW_B, opacity=0.92).set_stroke(YELLOW_B, width=2.0, opacity=1.0),
            FadeIn(close_race[2], shift=UP * 0.05),
            run_time=0.45,
        )
        self.wait(0.6)

        boundary_title = Tex(r"\text{The deeper issue is the learned decision boundary}", font_size=34, color=WHITE)
        boundary_title.move_to(title_anchor)
        boundary_caption = TexText("crowded regions make several names plausible", font_size=23, color=GREY_A)
        boundary_caption.move_to(caption_anchor)
        boundary = self.make_decision_boundary()
        boundary.move_to(ORIGIN + DOWN * 0.05)

        self.play(
            FadeOut(VGroup(close_race, race_arrow), shift=DOWN * 0.08),
            Transform(title, boundary_title),
            Transform(stage_caption, boundary_caption),
            FadeIn(boundary, shift=UP * 0.08),
            frame.animate.set_stroke(BLUE_B, width=1.25, opacity=0.95),
            run_time=1.0,
        )
        self.play(
            boundary[4].animate.set_stroke(YELLOW_B, width=2.8, opacity=1.0),
            run_time=0.55,
        )
        self.play(
            boundary[7].animate.shift(RIGHT * 0.45),
            *[
                dot.animate.shift(RIGHT * 0.45).set_fill(dot.get_color(), opacity=0.98).set_stroke(YELLOW_B, width=2.7, opacity=1.0).scale(1.18)
                for dot in boundary[8]
            ],
            boundary[6][3].animate.set_color(YELLOW_B),
            run_time=0.55,
        )
        self.wait(0.7)

        chart_title = Tex(r"\text{Balanced validation, uneven winners}", font_size=34, color=WHITE)
        chart_title.move_to(title_anchor)
        chart_caption = TexText("some names absorb errors from nearby names", font_size=23, color=GREY_A)
        chart_caption.move_to(caption_anchor)
        dist_names = ["Anna", "Chris", "Michael", "Sarah", "Robert", "David", "Paul", "long tail"]
        dist_start_values = [24, 22, 20, 18, 17, 15, 13, 11]
        dist_values = [44, 35, 28, 21, 4, 3, 2, 1]
        dist_colors = [YELLOW_B, GREEN_B, BLUE_B, TEAL_B, GREY_B, GREY_B, GREY_B, GREY_C]
        dist_chart = self.make_prediction_distribution("50-name run: top predictions", dist_names, dist_start_values, dist_colors)
        dist_chart.move_to(ORIGIN + DOWN * 0.55)
        dist_final_chart = self.make_prediction_distribution("50-name run: top predictions", dist_names, dist_values, dist_colors)
        dist_final_chart.move_to(dist_chart)

        self.play(
            FadeOut(boundary, shift=DOWN * 0.08),
            Transform(title, chart_title),
            Transform(stage_caption, chart_caption),
            FadeIn(dist_chart, shift=UP * 0.08),
            frame.animate.set_stroke(GREEN_B, width=1.25, opacity=0.95),
            run_time=1.0,
        )
        error_dots = VGroup()
        for index in range(24):
            source_bar = dist_chart[2][4 + index % 4]
            target_bar = dist_chart[2][index % 4]
            dot = Dot(
                source_bar.get_top() + UP * 0.08 + RIGHT * ((index % 3) - 1) * 0.045,
                radius=0.035,
            )
            dot.set_color(YELLOW_B if index % 2 == 0 else TEAL_B)
            dot.set_opacity(0.82)
            dot.generate_target()
            dot.target.move_to(target_bar.get_top() + UP * (0.16 + 0.025 * (index % 5)))
            dot.target.set_opacity(0.0)
            error_dots.add(dot)
        self.play(FadeIn(error_dots), run_time=0.2)
        self.play(
            Transform(dist_chart, dist_final_chart),
            LaggedStart(*[MoveToTarget(dot) for dot in error_dots], lag_ratio=0.018),
            run_time=1.15,
        )
        self.remove(error_dots)
        self.play(
            LaggedStart(*[
                bar.animate.set_fill(dist_colors[index], opacity=0.90 if index < 3 else 0.46)
                for index, bar in enumerate(dist_chart[2])
            ], lag_ratio=0.05),
            run_time=0.75,
        )
        self.wait(0.65)

        tail_title = Tex(r"\text{Scaling to 100+ names keeps the long tail}", font_size=34, color=WHITE)
        tail_title.move_to(title_anchor)
        tail_caption = TexText("attractor names change, but dominance remains", font_size=23, color=GREY_A)
        tail_caption.move_to(caption_anchor)
        tail_names = ["Ethan", "Noah", "Liam", "Mason", "Lucas", "Henry", "Owen", "Jack", "Ryan", "Cole", "Miles", "Adam", "tail"]
        tail_values = [70, 52, 34, 8, 7, 6, 5, 4, 3.4, 2.8, 2.2, 1.8, 1.2]
        tail_colors = [YELLOW_B, GREEN_B, BLUE_B, GREY_A, GREY_B, GREY_B, GREY_B, GREY_C, GREY_C, GREY_C, GREY_C, GREY_C, GREY_C]
        tail_chart = self.make_prediction_distribution("100+ names: few dominant winners", tail_names, tail_values, tail_colors)
        tail_chart.move_to(dist_chart)

        self.play(
            Transform(title, tail_title),
            Transform(stage_caption, tail_caption),
            FadeOut(dist_chart, shift=DOWN * 0.08),
            FadeIn(tail_chart, shift=UP * 0.08),
            run_time=0.95,
        )
        self.play(
            tail_chart[2][0].animate.set_fill(YELLOW_B, opacity=0.96).set_stroke(YELLOW_B, width=1.8, opacity=1.0),
            tail_chart[2][1].animate.set_fill(GREEN_B, opacity=0.90).set_stroke(GREEN_B, width=1.6, opacity=1.0),
            tail_chart[2][2].animate.set_fill(BLUE_B, opacity=0.86).set_stroke(BLUE_B, width=1.5, opacity=1.0),
            run_time=0.55,
        )
        self.wait(1.2)
