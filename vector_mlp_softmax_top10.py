from manimlib import *
import numpy as np


class VectorMLPSoftmaxTop10(Scene):
    LOGIT_VALUES = [-0.9, 0.3, 1.5, -0.4, 0.8, 2.2, -1.1, 0.15, 1.1, -0.7, 0.55]

    def create_embedding_vector(self):
        values = ["0.12", "-0.04", "0.37", "0.08", r"\cdots", "-0.21", "0.31"]
        entries = VGroup(*[
            Tex(value, font_size=28, color=GREY_A)
            for value in values
        ])
        entries.arrange(DOWN, buff=0.12)

        top_bracket = Tex(r"\overbrace{\phantom{000000000}}", font_size=36, color=WHITE)
        bottom_bracket = Tex(r"\underbrace{\phantom{000000000}}", font_size=36, color=WHITE)
        top_bracket.stretch_to_fit_width(entries.get_width() + 0.6)
        bottom_bracket.stretch_to_fit_width(entries.get_width() + 0.6)
        top_bracket.next_to(entries, UP, buff=0.08)
        bottom_bracket.next_to(entries, DOWN, buff=0.08)

        label = Tex(r"768\ \text{dimension}", font_size=24, color=GREY_B)
        label.next_to(bottom_bracket, DOWN, buff=0.16)

        vector = VGroup(top_bracket, entries, bottom_bracket, label)

        return vector

    def create_mlp(self):
        layer_sizes = [6, 4, 3]

        node_layers = VGroup()
        for layer_index, size in enumerate(layer_sizes):
            nodes = VGroup()
            for i in range(size):
                node = Circle(radius=0.105)
                node.set_fill(BLACK, opacity=0.0)
                node.set_stroke(WHITE, width=1.6, opacity=0.92)
                nodes.add(node)
            nodes.arrange(DOWN, buff=0.17)

            node_layers.add(nodes)
        node_layers.arrange(RIGHT, buff=0.62)

        connections = VGroup()
        for left_layer, right_layer in zip(node_layers[:-1], node_layers[1:]):
            for left_node in left_layer:
                for right_node in right_layer:
                    line = Line(left_node.get_center(), right_node.get_center())
                    line.set_stroke(GREY_B, width=0.85, opacity=0.28)
                    connections.add(line)

        return VGroup(connections, node_layers)

    def create_logit_numbers(self):
        numbers = VGroup()
        offsets = [
            LEFT * 0.03 + UP * 0.02,
            RIGHT * 0.02 + DOWN * 0.01,
            LEFT * 0.01 + DOWN * 0.03,
            RIGHT * 0.04 + UP * 0.01,
        ]
        for i, value in enumerate(self.LOGIT_VALUES):
            number = Tex(f"{value:+.2f}", font_size=20, color=GREY_A)
            number.shift(offsets[i % len(offsets)])
            numbers.add(number)
        return numbers

    def create_logits(self):
        values = self.LOGIT_VALUES
        n = len(values)
        max_abs = max(abs(v) for v in values)
        bars = VGroup()
        value_labels = VGroup()
        axis_height = 2.3
        row_spacing = axis_height / (n - 1)
        bar_height = row_spacing * 0.48
        max_bar_width = 1.1
        for i, value in enumerate(values):
            w = abs(value) / max_abs * max_bar_width
            bar = Rectangle(width=w, height=bar_height)
            bar.set_fill(WHITE if value >= 0 else GREY_D, opacity=0.84 if value >= 0 else 0.56)
            bar.set_stroke(WHITE, width=0.45, opacity=0.45)
            y = ((n - 1) / 2 - i) * row_spacing
            x = w / 2 if value >= 0 else -w / 2
            bar.move_to(np.array([x, y, 0.0]))
            bars.add(bar)

            label = Tex(f"{value:+.1f}", font_size=13, color=GREY_B)
            if value >= 0:
                label.next_to(bar, RIGHT, buff=0.06)
            else:
                label.next_to(bar, LEFT, buff=0.06)
            value_labels.add(label)

        baseline = Line(
            np.array([0, -axis_height / 2 - 0.12, 0]),
            np.array([0, axis_height / 2 + 0.12, 0]),
            stroke_color=WHITE,
            stroke_width=1.15,
        )
        return VGroup(baseline, bars, value_labels)

    def create_top10_from_bars(self, source_bars, order, probs):
        names = [
            "David",
            "Daniel",
            "Michael",
            "Kevin",
            "Jason",
            "Alex",
            "Chris",
            "Ryan",
            "Brian",
            "Eric",
        ]
        rows = VGroup()
        bar_left_x = -0.12
        name_x = -0.52
        value_x = 1.08
        row_gap = 0.26
        for rank, idx in enumerate(order):
            color = WHITE if rank == 0 else GREY_A
            name = Tex(r"\text{" + names[rank] + "}", font_size=16, color=color)
            value = Tex(f"{probs[int(idx)]:.2f}", font_size=15, color=color)
            bar = source_bars[int(idx)].copy()
            bar.set_fill(TEAL_C if rank == 0 else GREY_B, opacity=0.86 if rank == 0 else 0.55)
            bar.set_stroke(WHITE, width=0.25, opacity=0.35)
            bar.stretch(0.52, 0)
            bar.stretch_to_fit_height(0.07)
            y = -rank * row_gap
            name.move_to(np.array([name_x, y, 0.0]))
            bar.move_to(np.array([bar_left_x + bar.get_width() / 2, y, 0.0]))
            value.move_to(np.array([value_x, y, 0.0]))
            rows.add(VGroup(name, bar, value))

        subtitle = Tex(r"\text{Top 10 names}", font_size=19, color=GREY_B)
        subtitle.next_to(rows, DOWN, buff=0.22)
        return VGroup(rows, subtitle)

    def construct(self):
        self.camera.background_color = BLACK

        embedding = self.create_embedding_vector()
        embedding.move_to(LEFT * 2.9 + DOWN * 0.05)

        mlp = self.create_mlp()
        mlp.move_to(LEFT * 0.7 + DOWN * 0.05)

        logits = self.create_logits()
        logits.scale(0.84)
        logits.move_to(RIGHT * 2.0 + UP * 0.08)
        logit_numbers = self.create_logit_numbers()
        logit_numbers.move_to(mlp[1][-1].get_right() + RIGHT * 0.42)

        self.play(
            FadeIn(embedding, shift=RIGHT * 0.08),
            run_time=0.9,
        )
        self.wait(0.25)

        mlp_label = Tex(r"\text{MLP}", font_size=23, color=GREY_A)
        mlp_label.next_to(mlp, DOWN, buff=0.22)
        self.play(FadeIn(mlp, shift=UP * 0.12), FadeIn(mlp_label), run_time=0.7)
        entry_indices = [0, 1, 2, 3, 5, 6]
        input_nodes = mlp[1][0]
        value_clones = VGroup(*[
            embedding[1][index].copy()
            for index in entry_indices
        ])
        target_values = value_clones.copy()
        for value, node in zip(target_values, input_nodes):
            value.scale(0.34)
            value.move_to(node.get_center())
            value.set_opacity(0.0)

        self.play(
            LaggedStart(
                *[
                    Transform(clone, target)
                    for clone, target in zip(value_clones, target_values)
                ],
                lag_ratio=0.08,
            ),
            run_time=1.15,
            rate_func=smooth,
        )
        self.play(FadeOut(value_clones), run_time=0.18)

        node_layers = mlp[1]
        for i, layer in enumerate(node_layers):
            self.play(
                *[
                    node.animate.set_fill(WHITE, opacity=1.0).set_stroke(WHITE, width=2.0, opacity=1.0)
                    for node in layer
                ],
                run_time=0.38,
            )
            self.play(
                *[
                    node.animate.set_fill(BLACK, opacity=0.0).set_stroke(WHITE, width=1.6, opacity=0.92)
                    for node in layer
                ],
                run_time=0.25,
            )

        self.play(
            LaggedStart(
                *[FadeIn(number, shift=RIGHT * 0.04) for number in logit_numbers],
                lag_ratio=0.035,
            ),
            run_time=0.75,
        )
        self.play(ReplacementTransform(logit_numbers, logits), run_time=1.0, rate_func=smooth)

        softmax_frame = SurroundingRectangle(logits, buff=0.14)
        softmax_frame.set_fill(TEAL_C, opacity=0.035)
        softmax_frame.set_stroke(TEAL_C, width=0.8, opacity=0.9)
        softmax_label = Tex(r"\text{Softmax}", font_size=22, color=GREY_A)
        softmax_label.next_to(softmax_frame, DOWN, buff=0.18)
        softmax_label.set_y(mlp_label.get_y())
        self.play(ShowCreation(softmax_frame), FadeIn(softmax_label), run_time=0.45)

        probabilities = logits.copy()
        logit_array = np.array(self.LOGIT_VALUES)
        exp_values = np.exp(logit_array - np.max(logit_array))
        probs = exp_values / exp_values.sum()
        axis_x = probabilities[0].get_center()[0] - 0.72
        max_prob_width = 1.58
        probabilities[0].shift(RIGHT * (axis_x - probabilities[0].get_center()[0]))

        for bar, prob in zip(probabilities[1], probs):
            new_width = max_prob_width * prob / probs.max()
            bar.set_fill(TEAL_C, opacity=0.78)
            bar.stretch(new_width / bar.get_width(), 0)
            bar.move_to(np.array([axis_x + new_width / 2, bar.get_center()[1], 0.0]))
        probabilities[2].set_opacity(0.0)
        probabilities.move_to(logits)
        softmax_frame_target = SurroundingRectangle(probabilities, buff=0.14)
        softmax_frame_target.set_fill(TEAL_C, opacity=0.035)
        softmax_frame_target.set_stroke(TEAL_C, width=0.8, opacity=0.9)
        softmax_label_target = softmax_label.copy()
        softmax_label_target.next_to(softmax_frame_target, DOWN, buff=0.18)
        softmax_label_target.set_y(mlp_label.get_y())

        self.play(
            Transform(logits, probabilities),
            Transform(softmax_frame, softmax_frame_target),
            Transform(softmax_label, softmax_label_target),
            run_time=0.9,
        )
        order = np.argsort(-probs)[:10]
        top10 = self.create_top10_from_bars(logits[1], order, probs)
        top10.move_to(RIGHT * 4.75 + DOWN * 0.02)
        top10_rows = top10[0]
        top10_bars = VGroup(*[row[1] for row in top10_rows])
        top10_names = VGroup(*[row[0] for row in top10_rows])
        top10_values = VGroup(*[row[2] for row in top10_rows])
        self.play(
            LaggedStart(
                *[
                    TransformFromCopy(logits[1][int(idx)], top10_bars[i])
                    for i, idx in enumerate(order)
                ],
                lag_ratio=0.045,
            ),
            LaggedStart(*[FadeIn(name) for name in top10_names], lag_ratio=0.045),
            LaggedStart(*[FadeIn(value) for value in top10_values], lag_ratio=0.045),
            run_time=1.05,
        )
        self.play(FadeIn(top10[1]), run_time=0.35)
        self.wait(0.8)
