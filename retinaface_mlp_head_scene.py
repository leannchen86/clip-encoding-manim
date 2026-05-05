from manimlib import *
import numpy as np


class RetinaFaceMLPHeadScene(Scene):
    def make_cls_embedding_vector(self):
        return Tex(
            r"\left[n_1,\ n_2,\ \ldots,\ n_{768}\right]",
            font_size=42,
            color=YELLOW_B,
        )

    def make_numeric_embedding_vector(self):
        return Tex(
            r"\left[0.31,\ -0.22,\ 0.08,\ \ldots,\ 0.48\right]",
            font_size=40,
            color=YELLOW_B,
        )

    def make_embedding_space(self):
        axes = ThreeDAxes(
            x_range=(-1.2, 1.2, 10),
            y_range=(-1.2, 1.2, 10),
            z_range=(-1.2, 1.2, 10),
            width=1.9,
            height=1.9,
            depth=1.9,
        )
        axes.x_axis.set_stroke(BLUE_B, width=1.2, opacity=0.72)
        axes.y_axis.set_stroke(TEAL_B, width=1.2, opacity=0.72)
        axes.z_axis.set_stroke(YELLOW_B, width=1.2, opacity=0.72)

        sphere = Sphere(radius=1)
        sphere.set_color(BLUE_D)
        sphere.set_opacity(0.1)

        x_label = Tex("x", font_size=18, color=BLUE_B)
        x_label.move_to(RIGHT * 1.15 + DOWN * 0.06)
        y_label = Tex("y", font_size=18, color=TEAL_B)
        y_label.move_to(UP * 1.15 + RIGHT * 0.06)
        z_label = Tex("z", font_size=18, color=YELLOW_B)
        z_label.move_to(LEFT * 0.20 + UP * 0.20)

        space = Group(axes, sphere, x_label, y_label, z_label)
        space.rotate(45 * DEGREES, axis=OUT, about_point=ORIGIN)
        space.rotate(-45 * DEGREES, axis=RIGHT, about_point=ORIGIN)
        return space

    def make_embedding_vector(self, end, color=GREEN_B):
        vector = Line(ORIGIN, end, color=color, stroke_width=3.0)
        vector.set_opacity(1.0)
        endpoint = Sphere(radius=0.055)
        endpoint.set_color(color)
        endpoint.move_to(end)
        embedding_vector = Group(vector, endpoint)
        embedding_vector.rotate(45 * DEGREES, axis=OUT, about_point=ORIGIN)
        embedding_vector.rotate(-45 * DEGREES, axis=RIGHT, about_point=ORIGIN)
        return embedding_vector

    def make_weight_matrix(self):
        return Tex(
            r"\mathbf{W}="
            r"\left[\begin{array}{cccccc}"
            r"0.12&-0.08&0.05&\cdots&-0.02&0.03\\"
            r"-0.04&0.19&-0.07&\cdots&0.06&-0.11\\"
            r"0.09&0.01&0.14&\cdots&-0.05&0.02\\"
            r"-0.13&0.04&0.08&\cdots&0.10&-0.06\\"
            r"\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\"
            r"0.07&0.02&-0.10&\cdots&0.04&0.16"
            r"\end{array}\right]",
            font_size=27,
            color=WHITE,
        )

    def make_vertical_vector(self, values, color=YELLOW_B):
        return Tex(
            r"\left[\begin{array}{c}"
            + r"\\".join(values)
            + r"\end{array}\right]",
            font_size=30,
            color=color,
        )

    def make_fixed_width_vertical_vector(self, values, color=YELLOW_B, cell_width="0.90cm"):
        boxed_values = [
            rf"\makebox[{cell_width}][c]{{${value}$}}"
            for value in values
        ]
        return Tex(
            r"\left[\begin{array}{c}"
            + r"\\".join(boxed_values)
            + r"\end{array}\right]",
            font_size=30,
            color=color,
        )

    def make_matrix_annotations(self, matrix, name="Weight"):
        matrix_body = VGroup(*matrix[2:]) if len(matrix) > 2 else matrix
        height_brace = Brace(matrix, RIGHT, buff=0.06)
        height_brace.set_stroke(GREY_A, width=1.1, opacity=0.85)
        height_label = Tex("256", font_size=30, color=GREY_A)
        height_label.next_to(height_brace, RIGHT, buff=0.08)

        width_brace = Brace(matrix_body, DOWN, buff=0.07)
        width_brace.set_stroke(GREY_A, width=1.1, opacity=0.85)
        width_label = Tex("768", font_size=30, color=GREY_A)
        width_label.next_to(width_brace, DOWN, buff=0.08)

        name_label = Tex(rf"\text{{{name}}}", font_size=26, color=GREEN_B)
        name_label.next_to(width_label, DOWN, buff=0.10)
        return VGroup(height_brace, height_label, width_brace, width_label, name_label)

    def make_vector_annotations(self, vector, name=None, color=GREY_A, side=LEFT):
        height_brace = Brace(vector, side, buff=0.05)
        height_brace.set_stroke(color, width=1.0, opacity=0.82)
        height_label = Tex("256", font_size=28, color=color)
        height_label.next_to(height_brace, side, buff=0.08)
        labels = VGroup(height_brace, height_label)
        if name:
            name_label = Tex(rf"\text{{{name}}}", font_size=24, color=color)
            name_label.next_to(vector, DOWN, buff=0.16)
            labels.add(name_label)
        return labels

    def make_linear_block(self, label_text="Linear (Weight)", width=1.82):
        box = RoundedRectangle(
            width=width,
            height=0.48,
            corner_radius=0.09,
            stroke_color=GREEN_B,
            stroke_width=1.15,
            fill_color="#111820",
            fill_opacity=0.92,
        )
        label = TexText(label_text, font_size=14, color=WHITE)
        label.move_to(box)
        return VGroup(box, label)

    def make_batchnorm_param_vector(self, values, color=WHITE):
        return Tex(
            r"\left[\begin{array}{c}"
            + r"\\".join(values)
            + r"\end{array}\right]",
            font_size=25,
            color=color,
        )

    def make_training_batch_stack(self, center, index=0):
        color_sets = [
            [BLUE_B, TEAL_B, GREEN_B, YELLOW_B, GREY_A],
            [TEAL_C, BLUE_C, GREEN_C, YELLOW_C, GREY_B],
            [GREEN_B, BLUE_B, TEAL_B, YELLOW_B, GREY_A],
        ]
        value_sets = [
            ["0.24", "-0.13", r"\vdots", "0.31"],
            ["-0.05", "0.18", r"\vdots", "-0.02"],
            ["0.11", "0.02", r"\vdots", "0.19"],
            ["0.34", "-0.21", r"\vdots", "0.08"],
            ["-0.17", "0.09", r"\vdots", "0.22"],
        ]

        vectors = VGroup()
        colors = color_sets[index % len(color_sets)]
        opacities = [0.18, 0.24, 0.31, 0.40, 0.86]
        for i, values in enumerate(value_sets):
            vector = self.make_vertical_vector(values, color=colors[i])
            vector.set_opacity(opacities[i])
            vector.shift((LEFT + UP) * 0.075 * (len(value_sets) - 1 - i))
            vectors.add(vector)
        vectors.move_to(center)

        label = TexText("Training Batch", font_size=20, color=GREY_A)
        label.next_to(vectors, DOWN, buff=0.16)
        return VGroup(vectors, label)

    def make_bn_group_box(self, group, label_text, color):
        box = SurroundingRectangle(group, buff=0.16)
        box.set_fill(color, opacity=0.035)
        box.set_stroke(color, width=1.15, opacity=0.9)
        label = TexText(label_text, font_size=18, color=color)
        label.next_to(box, DOWN, buff=0.13)
        return VGroup(box, label)

    def make_two_layer_mlp(self):
        layer_sizes = [5, 4]
        layer_colors = [YELLOW_B, GREEN_B]

        node_layers = VGroup()
        for layer_index, size in enumerate(layer_sizes):
            nodes = VGroup()
            for _ in range(size):
                node = Circle(radius=0.095)
                node.set_fill(layer_colors[layer_index], opacity=0.12)
                node.set_stroke(layer_colors[layer_index], width=1.45, opacity=0.9)
                nodes.add(node)
            nodes.arrange(DOWN, buff=0.18)
            node_layers.add(nodes)
        node_layers.arrange(RIGHT, buff=1.0)

        connections = VGroup()
        for left_node in node_layers[0]:
            for right_node in node_layers[1]:
                line = Line(left_node.get_center(), right_node.get_center())
                line.set_stroke(GREY_B, width=0.85, opacity=0.34)
                connections.add(line)

        return VGroup(connections, node_layers)

    def construct(self):
        self.camera.background_color = BLACK

        clip_frame = RoundedRectangle(
            width=11.7,
            height=6.15,
            corner_radius=0.14,
            stroke_color=YELLOW_B,
            stroke_width=1.35,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        clip_frame.move_to(DOWN * 0.04)
        clip_title = Tex(r"\text{CLIP ViT-L/14 @ 336px}", font_size=36, color=WHITE)
        clip_title.move_to(clip_frame.get_top() + DOWN * 0.27)
        clip_subtitle = Tex(r"\text{(frozen, }\sim\text{ 300M params)}", font_size=22, color=GREY_A)
        clip_subtitle.next_to(clip_frame, DOWN, buff=0.12)

        projection_caption = Tex(
            r"\text{L2-normalize (rescale to unit length)}",
            font_size=25,
            color=GREEN_B,
        )
        projection_caption.move_to(clip_frame.get_bottom() + UP * 0.34)

        embedding_space = self.make_embedding_space()
        embedding_shift = UP * 0.24
        embedding_space.shift(embedding_shift)

        raw_endpoint = np.array([1.58, 0.82, 0.55])
        normalized_endpoint = raw_endpoint / np.linalg.norm(raw_endpoint)
        normalized_embedding_vector = self.make_embedding_vector(normalized_endpoint, color=GREEN_B)
        normalized_embedding_vector.shift(embedding_shift)
        normalized_vector_label = Tex(r"\hat{\mathbf{v}}", font_size=27, color=GREEN_B)
        normalized_vector_label.next_to(normalized_embedding_vector[1], UR, buff=0.12)

        self.add(
            clip_frame,
            clip_title,
            clip_subtitle,
            projection_caption,
            embedding_space,
            normalized_embedding_vector,
            normalized_vector_label,
        )
        self.wait(0.6)

        mlp_head_frame = RoundedRectangle(
            width=clip_frame.get_width(),
            height=clip_frame.get_height(),
            corner_radius=0.14,
            stroke_color=GREEN_B,
            stroke_width=1.25,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        mlp_head_frame.move_to(clip_frame)
        mlp_head_title = Tex(r"\text{Trained MLP head}", font_size=36, color=WHITE)
        mlp_head_title.move_to(mlp_head_frame.get_top() + DOWN * 0.27)
        mlp_head_subtitle = Tex(
            r"\text{248K params, the only thing that learned face}\rightarrow\text{name}",
            font_size=22,
            color=GREY_A,
        )
        mlp_head_subtitle.next_to(mlp_head_frame, DOWN, buff=0.12)

        final_embedding_vector = self.make_cls_embedding_vector()
        final_embedding_vector.move_to(ORIGIN + DOWN * 0.02)
        exiting_vit = VGroup(clip_frame, clip_title, clip_subtitle)
        embedding_world = Group(
            embedding_space,
            normalized_embedding_vector,
            normalized_vector_label,
        )

        self.play(
            exiting_vit.animate.scale(2.7, about_point=clip_frame.get_center()).set_opacity(0),
            FadeOut(embedding_world, shift=DOWN * 0.08),
            FadeOut(projection_caption, shift=DOWN * 0.05),
            FadeIn(final_embedding_vector, shift=UP * 0.08),
            FadeIn(mlp_head_frame),
            FadeIn(mlp_head_title, shift=UP * 0.06),
            FadeIn(mlp_head_subtitle, shift=UP * 0.04),
            run_time=1.2,
        )
        self.wait(1.25)

        # PART 2 -- First linear layer compresses 768 dims to 256 dims
        numeric_embedding_vector = self.make_numeric_embedding_vector()
        numeric_embedding_vector.move_to(final_embedding_vector)
        weight_matrix = self.make_weight_matrix()
        weight_matrix.move_to(RIGHT * 0.92 + UP * 0.08)
        weight_annotations = self.make_matrix_annotations(weight_matrix, name="Weight")
        weight_matrix_group = VGroup(weight_matrix, weight_annotations)
        mlp_network = self.make_two_layer_mlp()
        mlp_network.move_to(ORIGIN + UP * 0.15)
        compressed_vector = self.make_vertical_vector(
            ["0.18", "-0.07", r"\vdots", "0.26"],
            color=YELLOW_B,
        )
        compressed_vector.next_to(weight_matrix, LEFT, buff=0.86)
        compressed_vector.shift(UP * 0.08)
        compressed_annotations = self.make_vector_annotations(compressed_vector, side=LEFT)

        linear_block = self.make_linear_block("Linear (Weight)")
        linear_block.move_to(LEFT * 3.75 + DOWN * 1.48)
        bias_block = self.make_linear_block("Linear (Bias)")
        bias_block.next_to(linear_block, DOWN, buff=0.07)

        bias_vector = self.make_vertical_vector(
            ["0.03", "-0.02", r"\vdots", "0.01"],
            color=BLUE_B,
        )
        bias_vector.next_to(compressed_vector, RIGHT, buff=1.36)
        bias_annotations = self.make_vector_annotations(bias_vector, name="Bias", color=BLUE_B, side=LEFT)
        plus_sign = Tex(r"+", font_size=38, color=WHITE)
        plus_sign.move_to((compressed_vector.get_right() + bias_vector.get_left()) / 2 + LEFT * 0.3)

        biased_vector = self.make_vertical_vector(
            ["0.21", "-0.09", r"\vdots", "0.27"],
            color=GREEN_B,
        )
        biased_vector.move_to(compressed_vector)
        biased_annotations = self.make_vector_annotations(biased_vector, color=GREEN_B, side=LEFT)
        linear_caption = Tex(
            r"\text{Linear }(768\rightarrow256)",
            font_size=25,
            color=GREEN_B,
        )
        linear_caption.move_to(mlp_head_frame.get_bottom() + UP * 0.34)

        self.play(
            Transform(final_embedding_vector, numeric_embedding_vector),
            run_time=0.6,
        )
        self.play(
            final_embedding_vector.animate.move_to(ORIGIN + UP * 1.8),
            FadeIn(linear_caption, shift=UP * 0.05),
            run_time=0.85,
        )
        self.play(
            FadeIn(mlp_network, shift=UP * 0.08),
            run_time=0.65,
        )
        self.wait(0.35)
        self.play(
            Transform(mlp_network, weight_matrix_group),
            run_time=0.85,
        )
        self.play(
            final_embedding_vector.animate.move_to(weight_matrix.get_center() + UP * 0.05).scale(0.78),
            run_time=0.55,
        )
        self.play(
            FadeOut(final_embedding_vector, shift=DOWN * 0.15),
            FadeIn(compressed_vector, shift=RIGHT * 0.16),
            FadeIn(compressed_annotations, shift=RIGHT * 0.06),
            run_time=0.75,
        )
        self.play(
            FadeOut(mlp_network, scale=0.45, shift=LEFT * 0.9 + DOWN * 0.55),
            FadeIn(linear_block, shift=LEFT * 0.25 + DOWN * 0.08),
            run_time=0.75,
        )
        self.play(
            FadeIn(plus_sign, scale=0.75),
            FadeIn(bias_vector, shift=LEFT * 0.12),
            FadeIn(bias_annotations, shift=LEFT * 0.08),
            run_time=0.65,
        )
        bias_addend = bias_vector.copy()
        bias_addend_annotations = bias_annotations.copy()
        self.play(
            bias_addend.animate.move_to(compressed_vector).set_opacity(0.0),
            bias_addend_annotations.animate.move_to(compressed_annotations).set_opacity(0.0),
            ReplacementTransform(VGroup(bias_vector, bias_annotations), bias_block),
            FadeOut(plus_sign, shift=LEFT * 0.05),
            Transform(compressed_vector, biased_vector),
            Transform(compressed_annotations, biased_annotations),
            run_time=0.85,
        )
        self.wait(0.9)

        # PART 3 -- BatchNorm parameters operate on the 256-dim activation
        batchnorm_caption = Tex(
            r"\text{BatchNorm-1D }(256)",
            font_size=25,
            color=BLUE_B,
        )
        batchnorm_caption.move_to(linear_caption)

        param_specs = [
            (r"\text{running mean}", ["0", "0", r"\vdots", "0"], BLUE_B),
            (r"\text{running var}", ["0", "0", r"\vdots", "0"], TEAL_B),
            (r"\gamma", ["0.94", "1.08", r"\vdots", "0.87"], YELLOW_B),
            (r"\beta", ["-0.03", "0.12", r"\vdots", "0.05"], GREEN_B),
        ]

        bn_labels = VGroup()
        bn_vectors = VGroup()
        for label_tex, values, color in param_specs:
            label = Tex(label_tex, font_size=20, color=color)
            vector = self.make_vertical_vector(values, color=color)
            bn_labels.add(label)
            bn_vectors.add(vector)

        compressed_center = compressed_vector.get_center()
        bn_center_y = compressed_center[1]+ 1.85
        bn_label_centers = [
            np.array([compressed_center[0] + offset, bn_center_y, 0.0])
            for offset in (1.75, 3.00, 4.25, 5.50)
        ]
        for label, center in zip(bn_labels, bn_label_centers):
            label.move_to(center)

        for label, vector in zip(bn_labels, bn_vectors):
            vector.move_to(np.array([label.get_x(), bn_center_y, 0.0]))
        bn_label_targets = VGroup(*[
            label.copy().next_to(vector, DOWN, buff=0.14)
            for label, vector in zip(bn_labels, bn_vectors)
        ])

        self.play(
            Transform(linear_caption, batchnorm_caption),
            LaggedStart(*[
                FadeIn(label)
                for label in bn_labels
            ], lag_ratio=0.08),
            run_time=0.75,
        )
        self.play(
            LaggedStart(*[
                GrowFromCenter(vector)
                for vector in bn_vectors
            ], lag_ratio=0.08),
            Transform(bn_labels, bn_label_targets),
            run_time=0.85,
        )

        batch_center = np.array([
            (compressed_vector.get_x() + bn_vectors[0].get_x()) / 2 - 1,
            bn_center_y,
            0.0,
        ])
        training_batch = self.make_training_batch_stack(
            batch_center,
        )
        self.play(
            FadeIn(training_batch, shift=UP * 0.06),
            run_time=0.65,
        )

        statistics_group = VGroup(bn_vectors[0], bn_vectors[1], bn_labels[0], bn_labels[1])
        learned_group = VGroup(bn_vectors[2], bn_vectors[3], bn_labels[2], bn_labels[3])
        self.play(
            statistics_group.animate.shift(LEFT * 0.18),
            learned_group.animate.shift(RIGHT * 0.28),
            run_time=0.55,
        )
        statistics_box = self.make_bn_group_box(
            statistics_group,
            "Accumulated Statistics",
            BLUE_B,
        )
        learned_box = self.make_bn_group_box(
            learned_group,
            "Learned Parameters",
            GREEN_B,
        )
        self.play(
            FadeIn(statistics_box, scale=0.98),
            FadeIn(learned_box, scale=0.98),
            run_time=0.65,
        )

        bn_update_values = [
            [
                ["0.03", "-0.01", r"\vdots", "0.04"],
                ["0.91", "1.04", r"\vdots", "0.88"],
                ["0.96", "1.07", r"\vdots", "0.89"],
                ["-0.02", "0.10", r"\vdots", "0.06"],
            ],
            [
                ["0.05", "0.02", r"\vdots", "0.07"],
                ["0.86", "1.09", r"\vdots", "0.93"],
                ["0.98", "1.05", r"\vdots", "0.91"],
                ["-0.01", "0.08", r"\vdots", "0.08"],
            ],
            [
                ["0.08", "0.01", r"\vdots", "0.10"],
                ["0.82", "1.12", r"\vdots", "0.97"],
                ["1.01", "1.03", r"\vdots", "0.94"],
                ["0.00", "0.06", r"\vdots", "0.10"],
            ],
        ]
        bn_colors = [color for _, _, color in param_specs]
        update_target = VGroup(statistics_box, learned_box).get_center()
        for update_index, update_values in enumerate(bn_update_values):
            if update_index > 0:
                training_batch = self.make_training_batch_stack(
                    batch_center,
                    index=update_index,
                )
                self.play(
                    FadeIn(training_batch, shift=UP * 0.05),
                    run_time=0.45,
                )

            updated_vectors = VGroup()
            for vector, values, color in zip(bn_vectors, update_values, bn_colors):
                updated_vector = self.make_vertical_vector(values, color=color)
                updated_vector.move_to(vector)
                updated_vectors.add(updated_vector)

            self.play(
                training_batch.animate.move_to(update_target).scale(0.72).set_opacity(0.0),
                statistics_box[0].animate.set_stroke(BLUE_B, width=1.75, opacity=1.0),
                learned_box[0].animate.set_stroke(GREEN_B, width=1.75, opacity=1.0),
                run_time=0.7,
            )
            self.remove(training_batch)
            self.play(
                LaggedStart(*[
                    Transform(vector, updated_vector)
                    for vector, updated_vector in zip(bn_vectors, updated_vectors)
                ], lag_ratio=0.08),
                statistics_box[0].animate.set_stroke(BLUE_B, width=1.15, opacity=0.9),
                learned_box[0].animate.set_stroke(GREEN_B, width=1.15, opacity=0.9),
                run_time=0.75,
            )

        compressed_context = VGroup(compressed_vector, compressed_annotations)
        activation_anchor = compressed_vector.get_center()
        activation_left_x = compressed_vector.get_left()[0]
        parameter_fly_target = activation_anchor + UP * 2 + LEFT * 2.0
        caption_position = np.array([
            compressed_vector.get_x(),
            compressed_context.get_top()[1] + 0.34,
            0.0,
        ])

        def place_activation_vector(vector):
            vector.move_to(activation_anchor)
            vector.shift(RIGHT * (activation_left_x - vector.get_left()[0]))
            return vector

        normalization_caption = Tex(r"\text{Normalization}", font_size=28, color=BLUE_B)
        normalization_caption.move_to(caption_position + LEFT * 2.2)
        normalized_vector = self.make_fixed_width_vertical_vector(
            ["0.15", "-0.10", r"\vdots", "0.22"],
            color=BLUE_B,
        )
        place_activation_vector(normalized_vector)
        normalized_vector.shift(LEFT * 0.61 + UP * 1.88)
        normalization_copies = VGroup(bn_vectors[0].copy(), bn_vectors[1].copy())
        for copy in normalization_copies:
            copy.generate_target()
            copy.target.move_to(parameter_fly_target)
            copy.target.set_opacity(0.0)
        self.add(normalization_copies)
        self.play(
            FadeIn(normalization_caption),
            *[
                MoveToTarget(copy)
                for copy in normalization_copies
            ],
            FadeOut(compressed_vector),
            FadeIn(normalized_vector),
            run_time=0.85,
        )
        compressed_vector = normalized_vector
        self.remove(normalization_copies)
        self.play(
            FadeOut(normalization_caption),
            run_time=0.3,
        )

        affine_caption = Tex(
            r"\begin{array}{c}\text{Affine}\\\text{Transformation}\end{array}",
            font_size=28,
            color=GREEN_B,
        )
        affine_caption.move_to(caption_position + LEFT * 2.2 + UP * 0.1)
        affine_vector = self.make_fixed_width_vertical_vector(
            ["0.12", "0.04", r"\vdots", "0.31"],
            color=GREEN_B,
        )
        place_activation_vector(affine_vector)
        affine_vector.shift(LEFT * 0.61 + UP * 1.88)
        affine_copies = VGroup(bn_vectors[2].copy(), bn_vectors[3].copy())
        for copy in affine_copies:
            copy.generate_target()
            copy.target.move_to(parameter_fly_target)
            copy.target.set_opacity(0.0)
        self.add(affine_copies)
        self.play(
            FadeIn(affine_caption),
            *[
                MoveToTarget(copy)
                for copy in affine_copies
            ],
            FadeOut(compressed_vector),
            FadeIn(affine_vector),
            run_time=0.85,
        )
        compressed_vector = affine_vector
        self.remove(affine_copies)
        self.play(
            FadeOut(affine_caption),
            run_time=0.3,
        )

        batchnorm_block = self.make_linear_block("BatchNorm-1D (256)", width=2.25)
        batchnorm_block.next_to(VGroup(linear_block, bias_block), RIGHT, buff=0.18)
        batchnorm_objects = VGroup(
            bn_vectors,
            bn_labels,
            statistics_box,
            learned_box,
        )
        self.play(
            ReplacementTransform(batchnorm_objects, batchnorm_block),
            run_time=0.9,
        )
        self.wait(1.25)
