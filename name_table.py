from manimlib import *
import numpy as np

CLIP_EMBED_DIM = 768
HEAD_HIDDEN_DIM = 1024
GENERIC_HIDDEN_DIM_TEX = r"\text{hidden}"
CLASS_COUNT_SYMBOL = "C"


class NameTable(Scene):
    def make_linear_block(self, label_text="Linear (Weight)", width=1.82, font_size=14):
        box = RoundedRectangle(
            width=width,
            height=0.48,
            corner_radius=0.09,
            stroke_color=GREEN_B,
            stroke_width=1.15,
            fill_color="#111820",
            fill_opacity=0.92,
        )
        label = TexText(label_text, font_size=font_size, color=WHITE)
        label.move_to(box)
        return VGroup(box, label)

    def make_dimension_linear_block(self, in_dim, out_dim):
        box = RoundedRectangle(
            width=1.82,
            height=0.78,
            corner_radius=0.09,
            stroke_color=GREEN_B,
            stroke_width=1.15,
            fill_color="#111820",
            fill_opacity=0.92,
        )
        label = Tex(
            rf"\begin{{array}}{{c}}\text{{Linear}}\\{in_dim}\rightarrow {out_dim}\end{{array}}",
            font_size=17,
            color=WHITE,
        )
        label.move_to(box)
        return VGroup(box, label)

    def make_logits_block(self):
        box = RoundedRectangle(
            width=1.52,
            height=0.78,
            corner_radius=0.09,
            stroke_color=GREEN_B,
            stroke_width=1.15,
            fill_color="#111820",
            fill_opacity=0.92,
        )
        label = Tex(
            rf"\begin{{array}}{{c}}{CLASS_COUNT_SYMBOL}\ \text{{logits}}\\\text{{(real-valued)}}\end{{array}}",
            font_size=16,
            color=WHITE,
        )
        label.move_to(box)
        return VGroup(box, label)

    def make_relu_block(self):
        box = RoundedRectangle(
            width=0.78,
            height=0.78,
            corner_radius=0.08,
            stroke_color=YELLOW_B,
            stroke_width=1.15,
            fill_color="#111820",
            fill_opacity=0.94,
        )

        graph_origin = box.get_center() + UP * 0.10
        x_axis = Line(graph_origin + LEFT * 0.22, graph_origin + RIGHT * 0.24)
        y_axis = Line(graph_origin + DOWN * 0.12, graph_origin + UP * 0.22)
        axes = VGroup(x_axis, y_axis)
        axes.set_stroke(GREY_A, width=0.75, opacity=0.6)

        flat = Line(graph_origin + LEFT * 0.19, graph_origin)
        rising = Line(graph_origin, graph_origin + RIGHT * 0.22 + UP * 0.17)
        relu_curve = VGroup(flat, rising)
        relu_curve.set_stroke(YELLOW_B, width=1.6, opacity=0.96)

        label = TexText("ReLU", font_size=12, color=WHITE)
        label.move_to(box.get_center() + DOWN * 0.22)
        return VGroup(box, axes, relu_curve, label)

    def make_two_layer_mlp(self, layer_sizes=(4, 6)):
        node_layers = VGroup()
        for size in layer_sizes:
            nodes = VGroup()
            for _ in range(size):
                node = Circle(radius=0.095)
                node.set_fill(WHITE, opacity=0.0)
                node.set_stroke(WHITE, width=1.45, opacity=0.85)
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

    def make_horizontal_vector(self, values, color=GREEN_B):
        return Tex(
            r"\left["
            + r",\ ".join(values)
            + r"\right]",
            font_size=34,
            color=color,
        )

    def make_column(self, entries, color=WHITE, font_size=25, use_tex=True):
        column = VGroup()
        for entry in entries:
            item = Tex(entry, font_size=font_size, color=color) if use_tex else TexText(
                entry,
                font_size=font_size,
                color=color,
            )
            column.add(item)
        column.arrange(DOWN, buff=0.17)
        return column

    def make_param_note(self):
        return TexText(
            "Trainable params depend on hidden width and class count",
            font_size=22,
            color=WHITE,
        )

    def construct(self):
        self.camera.background_color = BLACK

        mlp_head_frame = RoundedRectangle(
            width=11.7,
            height=6.15,
            corner_radius=0.14,
            stroke_color=GREEN_B,
            stroke_width=1.25,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        mlp_head_frame.move_to(DOWN * 0.04)

        mlp_head_title = Tex(r"\text{Trained MLP head}", font_size=36, color=WHITE)
        mlp_head_title.move_to(mlp_head_frame.get_top() + DOWN * 0.27)
        mlp_head_subtitle = Tex(
            r"\text{deployed probe head}",
            font_size=22,
            color=GREY_A,
        )
        mlp_head_subtitle.next_to(mlp_head_frame, DOWN, buff=0.12)

        linear_block = self.make_linear_block("Linear (Weight)")
        linear_block.move_to(LEFT * 3.75 + DOWN * 1.48)
        bias_block = self.make_linear_block("Linear (Bias)")
        bias_block.next_to(linear_block, DOWN, buff=0.07)

        compact_batchnorm_block = self.make_linear_block(
            f"BatchNorm-1D ({HEAD_HIDDEN_DIM})",
            width=1.82,
        )
        compact_batchnorm_block.next_to(VGroup(linear_block, bias_block), RIGHT, buff=0.18)
        compact_batchnorm_block.shift(LEFT * 0.12)

        relu_block = self.make_relu_block()
        relu_block.next_to(compact_batchnorm_block, RIGHT, buff=0.07)

        merged_linear_block = self.make_dimension_linear_block(CLIP_EMBED_DIM, HEAD_HIDDEN_DIM)
        merged_linear_block.move_to(VGroup(linear_block, bias_block).get_center())

        second_linear_block = self.make_dimension_linear_block(HEAD_HIDDEN_DIM, GENERIC_HIDDEN_DIM_TEX)
        second_linear_block.next_to(relu_block, RIGHT, buff=0.07)

        batchnorm128_block = self.make_linear_block(
            "BatchNorm-1D",
            width=1.82,
            font_size=12,
        )
        batchnorm128_block.next_to(second_linear_block, RIGHT, buff=0.07)

        relu128_block = self.make_relu_block()
        relu128_block.next_to(batchnorm128_block, RIGHT, buff=0.07)

        final_linear_block = self.make_dimension_linear_block(GENERIC_HIDDEN_DIM_TEX, CLASS_COUNT_SYMBOL)
        final_linear_block.next_to(relu_block, DOWN, buff=0.12)
        final_linear_block.align_to(relu_block, RIGHT)
        final_linear_block.shift(UP * 0.08)

        logits_block = self.make_logits_block()
        logits_block.next_to(final_linear_block, RIGHT, buff=0.07)

        final_input_vector = self.make_horizontal_vector(
            ["0.95", "0", "0.22", r"\cdots", "0.63", "0.04"],
            color=YELLOW_B,
        )
        final_input_vector.move_to(ORIGIN + DOWN * 0.37)

        final_mlp_network = self.make_two_layer_mlp(layer_sizes=(4, 6))
        final_mlp_network.scale(0.94)
        final_mlp_network.next_to(final_input_vector, UP, buff=0.42)
        final_mlp_center = final_mlp_network.get_center()

        output138_vector = self.make_horizontal_vector(
            ["0.18", "-0.04", "0.33", "0.09", r"\cdots", "-0.12", "0.27", "0.05"],
            color=GREEN_B,
        )
        output138_vector.move_to(final_input_vector)
        output138_vector.shift(DOWN * 0.18)
        output138_vector.move_to(np.array([
            output138_vector.get_center()[0],
            final_mlp_center[1],
            0.0,
        ]))

        pipeline_group = VGroup(
            merged_linear_block,
            compact_batchnorm_block,
            relu_block,
            second_linear_block,
            batchnorm128_block,
            relu128_block,
            final_linear_block,
            logits_block,
        )
        param_note = self.make_param_note()
        param_note.move_to(np.array([
            pipeline_group.get_center()[0],
            pipeline_group.get_top()[1] + 0.82,
            0.0,
        ]))

        self.add(
            mlp_head_frame,
            mlp_head_title,
            mlp_head_subtitle,
            output138_vector,
            pipeline_group,
            param_note,
        )
        self.wait(1.0)

        logits_output_frame = RoundedRectangle(
            width=11.7,
            height=6.15,
            corner_radius=0.14,
            stroke_color=GREEN_B,
            stroke_width=1.25,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        logits_output_frame.move_to(mlp_head_frame)

        output_title = Tex(
            r"\text{Final Linear Layer Output}",
            font_size=36,
            color=WHITE,
        )
        output_title.move_to(logits_output_frame.get_top() + DOWN * 0.27)

        output_subtitle = TexText(
            "class-count vector of logits",
            font_size=24,
            color=GREY_A,
        )
        output_subtitle.next_to(logits_output_frame, DOWN, buff=0.12)

        old_frame_target = mlp_head_frame.copy()
        old_frame_target.scale(1.08)
        old_frame_target.set_opacity(0.0)

        pipeline_without_logits = VGroup(
            merged_linear_block,
            compact_batchnorm_block,
            relu_block,
            second_linear_block,
            batchnorm128_block,
            relu128_block,
            final_linear_block,
        )
        self.play(
            Transform(mlp_head_frame, old_frame_target),
            FadeOut(mlp_head_title, shift=UP * 0.05),
            FadeOut(mlp_head_subtitle, shift=DOWN * 0.04),
            FadeOut(param_note, shift=UP * 0.04),
            FadeOut(pipeline_without_logits, shift=DOWN * 0.08),
            FadeOut(logits_block[1], shift=DOWN * 0.03),
            Transform(logits_block[0], logits_output_frame),
            FadeIn(output_title, shift=DOWN * 0.04),
            FadeIn(output_subtitle, shift=UP * 0.04),
            run_time=1.05,
        )
        self.wait(0.8)

        index_column = self.make_column(
            ["0", "1", "2", "3", "4", "5", r"\vdots", "C-1"],
            color=GREY_A,
            font_size=25,
        )
        logit_column = self.make_column(
            ["0.18", "-0.04", "0.33", "0.09", "-0.12", "0.27", r"\vdots", "0.05"],
            color=GREEN_B,
            font_size=25,
        )
        name_entries = ["David", "Jennifer", "Brian", "John", "Paul", "Christian", r"\vdots", "Ted"]
        name_column = VGroup()
        for name_entry in name_entries:
            if name_entry == r"\vdots":
                name_item = Tex(name_entry, font_size=25, color=WHITE)
            else:
                name_item = TexText(name_entry, font_size=25, color=WHITE)
            name_column.add(name_item)
        name_column.arrange(DOWN, buff=0.17)

        table_center = logits_block[0].get_center() + LEFT * 0.55 + UP * 0.18
        logit_column.move_to(table_center)
        index_column.next_to(logit_column, LEFT, buff=2.0)
        index_column.align_to(logit_column, UP)
        name_column.next_to(logit_column, RIGHT, buff=2.5)
        name_column.align_to(logit_column, UP)

        index_title = TexText("Index", font_size=23, color=GREY_A)
        logit_title = TexText("Logit", font_size=23, color=GREEN_B)
        name_title = TexText("Name (from saved list)", font_size=23, color=WHITE)
        index_title.next_to(index_column, UP, buff=0.26)
        logit_title.next_to(logit_column, UP, buff=0.26)
        name_title.next_to(name_column, UP, buff=0.26)

        self.play(
            Transform(output138_vector, logit_column),
            FadeIn(logit_title, shift=UP * 0.04),
            run_time=0.62,
        )
        self.play(
            FadeIn(index_title, shift=UP * 0.04),
            FadeIn(name_title, shift=UP * 0.04),
            run_time=0.15,
        )
        for index_item, name_item in zip(index_column, name_column):
            self.play(
                FadeIn(index_item, shift=DOWN * 0.04),
                FadeIn(name_item, shift=DOWN * 0.04),
                run_time=0.04,
            )
        self.wait(0.8)

        saved_names_box = RoundedRectangle(
            width=6.25,
            height=0.62,
            corner_radius=0.08,
            stroke_color=BLUE_B,
            stroke_width=1.2,
            fill_color="#111820",
            fill_opacity=0.92,
        )
        saved_names_box.move_to(logits_block[0].get_bottom() + UP * 0.86)
        saved_names_text = TexText(
            'names = ["David", "Jennifer", "Brian", "John", ...]',
            font_size=20,
            color=WHITE,
        )
        saved_names_text.move_to(saved_names_box)
        saved_names_group = VGroup(saved_names_box, saved_names_text)

        saved_names_caption = Tex(
            r"\textit{names}\ \text{saved alongside the checkpoint, fixed at training time}",
            font_size=20,
            color=GREY_A,
        )
        saved_names_caption.next_to(saved_names_box, DOWN, buff=0.16)

        self.play(
            TransformFromCopy(VGroup(name_title, name_column), saved_names_group),
            FadeIn(saved_names_caption, shift=UP * 0.04),
            run_time=0.85,
        )
        self.wait(0.8)

        david_subtitle = TexText(
            "The model has never seen the string 'David.' It learned that index 0 should fire on David-faces.",
            font_size=20,
            color=GREY_A,
        )
        david_subtitle.next_to(logits_block[0], DOWN, buff=0.12)
        david_subtitle.scale(min(1.0, 11.35 / david_subtitle.get_width()))

        logit_column_swapped = logit_column.copy()
        logit_column_swapped.move_to(index_column.get_center())
        index_column_swapped = index_column.copy()
        index_column_swapped.move_to(logit_column.get_center())
        name_column_colored = name_column.copy()

        row_colors = [
            "#8AB4D6",
            "#A8C686",
            "#D8B46A",
            "#B89AD6",
            "#78C2B3",
            "#D99A7E",
            "#A9B7C9",
            "#C7A7B8",
        ]
        for row_index, row_color in enumerate(row_colors):
            logit_column_swapped[row_index].set_color(row_color)
            index_column_swapped[row_index].set_color(row_color)
            name_column_colored[row_index].set_color(row_color)

        logit_title_swapped = logit_title.copy()
        logit_title_swapped.next_to(logit_column_swapped, UP, buff=0.26)
        index_title_swapped = index_title.copy()
        index_title_swapped.next_to(index_column_swapped, UP, buff=0.26)

        self.play(
            Transform(output_subtitle, david_subtitle),
            Transform(output138_vector, logit_column_swapped),
            Transform(index_column, index_column_swapped),
            Transform(name_column, name_column_colored),
            Transform(logit_title, logit_title_swapped),
            Transform(index_title, index_title_swapped),
            run_time=0.85,
        )

        index_name_arrows = VGroup()
        arrow_x = (index_column.get_right()[0] + name_column.get_left()[0]) / 2
        for index_item, name_item in zip(index_column, name_column):
            arrow = Tex(r"\longleftarrow", font_size=24, color=BLUE_B)
            arrow.stretch(1.25, dim=0)
            arrow.move_to(np.array([
                arrow_x,
                (index_item.get_center()[1] + name_item.get_center()[1]) / 2,
                0.0,
            ]))
            index_name_arrows.add(arrow)

        for arrow in index_name_arrows:
            self.play(
                FadeIn(arrow, shift=DOWN * 0.03),
                run_time=0.06,
            )
        self.wait(0.45)

        mapping_subtitle = TexText(
            r"Name $\rightarrow$ index mapping is fixed at training time. Reorder the list, and the classifier must be retrained.",
            font_size=20,
            color=GREY_A,
        )
        mapping_subtitle.next_to(logits_block[0], DOWN, buff=0.12)
        mapping_subtitle.scale(min(1.0, 11.35 / mapping_subtitle.get_width()))

        shuffled_name_entries = ["Christian", "Brian", "Ted", "David", "Jennifer", "Paul", r"\vdots", "John"]
        name_row_centers = [item.get_center() for item in name_column]
        name_items_by_entry = {
            name_entry: name_column[index]
            for index, name_entry in enumerate(name_entries)
        }
        name_reorder_animations = []
        for row_index, name_entry in enumerate(shuffled_name_entries):
            name_item = name_items_by_entry[name_entry]
            name_item.generate_target()
            name_item.target.move_to(name_row_centers[row_index])
            name_reorder_animations.append(MoveToTarget(name_item))

        self.play(
            Transform(output_subtitle, mapping_subtitle),
            *name_reorder_animations,
            index_name_arrows.animate.set_color(RED_B),
            run_time=0.9,
        )
        self.wait(0.8)

        restored_name_column = VGroup()
        for row_index, name_entry in enumerate(name_entries):
            if name_entry == r"\vdots":
                name_item = Tex(name_entry, font_size=25, color=row_colors[row_index])
            else:
                name_item = TexText(name_entry, font_size=25, color=row_colors[row_index])
            restored_name_column.add(name_item)
        restored_name_column.arrange(DOWN, buff=0.17)
        restored_name_column.move_to(name_column)
        self.play(
            Transform(name_column, restored_name_column),
            output138_vector.animate.set_color(WHITE),
            index_column.animate.set_color(WHITE),
            index_name_arrows.animate.set_color(WHITE),
            run_time=0.75,
        )
        self.wait(0.25)

        softmax_title = Tex(
            r"\text{Softmax: Logits}\rightarrow\text{Probabilities}",
            font_size=36,
            color=WHITE,
        )
        softmax_frame = RoundedRectangle(
            width=11.7,
            height=6.15,
            corner_radius=0.14,
            stroke_color=BLUE_B,
            stroke_width=1.25,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        softmax_frame.move_to(logits_block[0])
        softmax_title.move_to(softmax_frame.get_top() + DOWN * 0.27)

        old_logits_frame_target = logits_block[0].copy()
        old_logits_frame_target.scale(1.08)
        old_logits_frame_target.set_opacity(0.0)

        final_name_column = name_column.copy()
        final_name_column.set_color(WHITE)
        final_name_column.arrange(DOWN, buff=0.17, aligned_edge=LEFT)
        final_name_column.move_to(logits_block[0].get_center() + LEFT * 0.78 + DOWN * 0.02)

        final_logit_column = logit_column_swapped.copy()
        final_logit_column.set_color(WHITE)
        final_logit_column.arrange(DOWN, buff=0.17, aligned_edge=LEFT)
        final_logit_column.next_to(final_name_column, RIGHT, buff=0.7)
        final_logit_column.align_to(final_name_column, UP)

        shared_logit_title = TexText(
            "C logits (real-valued)",
            font_size=25,
            color=WHITE,
        )
        final_columns_group = VGroup(final_name_column, final_logit_column)
        shared_logit_title.move_to(np.array([
            final_columns_group.get_center()[0],
            final_columns_group.get_top()[1] + 0.36,
            0.0,
        ]))

        self.play(
            Transform(logits_block[0], old_logits_frame_target),
            FadeIn(softmax_frame, scale=0.98),
            FadeOut(output_title, shift=UP * 0.05),
            FadeOut(output_subtitle, shift=DOWN * 0.04),
            FadeOut(index_column, shift=LEFT * 0.08),
            FadeOut(index_title, shift=UP * 0.04),
            FadeOut(name_title, shift=UP * 0.04),
            FadeOut(index_name_arrows),
            FadeOut(saved_names_group, shift=DOWN * 0.05),
            FadeOut(saved_names_caption, shift=DOWN * 0.04),
            Transform(name_column, final_name_column),
            Transform(output138_vector, final_logit_column),
            Transform(logit_title, shared_logit_title),
            FadeIn(softmax_title, shift=DOWN * 0.04),
            run_time=1.05,
        )
        self.wait(0.8)

        softmax_input_group = VGroup(logit_title, name_column, output138_vector)
        softmax_operator_label = Tex(r"\text{Softmax}", font_size=27, color=BLUE_B)
        softmax_formula = Tex(
            r"p_i=\frac{\exp(z_i)}{\sum_j \exp(z_j)}",
            font_size=28,
            color=WHITE,
        )
        softmax_operator_group = VGroup(softmax_operator_label, softmax_formula)
        softmax_operator_group.arrange(DOWN, buff=0.18)
        softmax_operator_group.move_to(LEFT * 0.22 + UP * 0.03)

        self.play(
            softmax_input_group.animate.shift(LEFT * 3.08),
            FadeIn(softmax_operator_label, shift=UP * 0.04),
            FadeIn(softmax_formula, shift=DOWN * 0.04),
            run_time=0.8,
        )
        self.wait(0.25)

        probability_names = VGroup()
        for name_entry in name_entries:
            if name_entry == r"\vdots":
                name_item = Tex(name_entry, font_size=25, color=WHITE)
            else:
                name_item = TexText(name_entry, font_size=25, color=WHITE)
            probability_names.add(name_item)
        probability_names.arrange(DOWN, buff=0.17, aligned_edge=LEFT)

        probability_values = self.make_column(
            ["0.063", "0.036", "0.094", "0.041", "0.029", "0.071", r"\vdots", "0.052"],
            color=TEAL_B,
            font_size=25,
        )
        probability_values.arrange(DOWN, buff=0.17, aligned_edge=LEFT)
        probability_values.next_to(probability_names, RIGHT, buff=0.7)
        probability_values.align_to(probability_names, UP)

        total_name = TexText(r"\textbf{Total}", font_size=25, color=WHITE)
        total_value = Tex(r"\mathbf{1.0}", font_size=25, color=TEAL_B)
        total_name.next_to(probability_names, DOWN, buff=0.28)
        total_name.align_to(probability_names, LEFT)
        total_value.move_to(np.array([
            probability_values.get_left()[0] + total_value.get_width() / 2,
            total_name.get_center()[1],
            0.0,
        ]))

        probability_title = TexText(
            "C probabilities (sum to 1.0)",
            font_size=25,
            color=WHITE,
        )
        probability_table = VGroup(
            probability_names,
            probability_values,
            total_name,
            total_value,
        )
        probability_table.move_to(RIGHT * 2.78 + DOWN * 0.02)
        probability_title.next_to(probability_table, UP, buff=0.36)
        probability_output_group = VGroup(probability_title, probability_table)
        probability_output_group.shift(UP * (logit_title.get_center()[1] - probability_title.get_center()[1]))
        probability_table.shift(UP * (name_column.get_top()[1] - probability_names.get_top()[1]))

        travelling_logits = softmax_input_group.copy()
        self.add(travelling_logits)
        squeezed_logits = travelling_logits.copy()
        squeezed_logits.scale(0.22)
        squeezed_logits.move_to(softmax_operator_group.get_center())
        squeezed_logits.set_opacity(0.58)
        self.play(
            Transform(travelling_logits, squeezed_logits),
            softmax_operator_label.animate.set_color(TEAL_B).scale(1.08),
            softmax_formula.animate.set_color(TEAL_B).scale(1.04),
            run_time=0.5,
        )
        self.play(
            Transform(travelling_logits, probability_output_group),
            softmax_operator_label.animate.set_color(BLUE_B).scale(1 / 1.08),
            softmax_formula.animate.set_color(WHITE).scale(1 / 1.04),
            run_time=0.9,
        )
        self.remove(travelling_logits)
        self.add(probability_output_group)
        self.play(
            probability_table.animate.shift(LEFT * 0.28),
            run_time=0.35,
        )

        probability_bar_values = [0.063, 0.036, 0.094, 0.041, 0.029, 0.071, None, 0.052]
        prob_bar_tracks = VGroup()
        prob_bar_fills = VGroup()
        bar_left_x = probability_values.get_right()[0] + 0.24
        max_bar_width = 0.86
        for prob_item, prob_value in zip(probability_values, probability_bar_values):
            if prob_value is None:
                continue
            track = RoundedRectangle(
                width=max_bar_width,
                height=0.13,
                corner_radius=0.045,
                stroke_color="#24464d",
                stroke_width=0.55,
                fill_color="#071419",
                fill_opacity=0.84,
            )
            track.move_to(np.array([
                bar_left_x + max_bar_width / 2,
                prob_item.get_center()[1],
                0.0,
            ]))
            bar_width = max_bar_width * prob_value / 0.1
            fill_bar = RoundedRectangle(
                width=bar_width,
                height=0.13,
                corner_radius=0.045,
                stroke_color=TEAL_B,
                stroke_width=0.35,
                fill_color=TEAL_B,
                fill_opacity=0.76,
            )
            fill_bar.move_to(np.array([
                bar_left_x + bar_width / 2,
                prob_item.get_center()[1],
                0.0,
            ]))
            prob_bar_tracks.add(track)
            prob_bar_fills.add(fill_bar)

        self.play(
            FadeIn(prob_bar_tracks, shift=RIGHT * 0.04),
            run_time=0.3,
        )
        self.play(
            LaggedStart(
                *[GrowFromEdge(bar, LEFT) for bar in prob_bar_fills],
                lag_ratio=0.08,
            ),
            run_time=0.85,
        )
        self.wait(0.35)

        sorted_probability_rows = [
            ("Brian", "0.094", 0.094),
            ("Christian", "0.071", 0.071),
            ("David", "0.063", 0.063),
            ("Ted", "0.052", 0.052),
            ("John", "0.041", 0.041),
            ("Jennifer", "0.036", 0.036),
            ("Paul", "0.029", 0.029),
            (r"\vdots", r"\vdots", None),
        ]
        ranked_rows = VGroup()
        ranked_max_bar_width = 1.05
        name_left_x = -1.85
        bar_left_x_target = -0.25
        value_left_x = bar_left_x_target + ranked_max_bar_width + 0.18
        for row_index, (name_entry, probability_text, probability_value) in enumerate(sorted_probability_rows):
            row_y = -0.38 * row_index
            if name_entry == r"\vdots":
                ranked_name = Tex(name_entry, font_size=24, color=WHITE)
            else:
                ranked_name = TexText(name_entry, font_size=24, color=WHITE)
            ranked_name.move_to(np.array([
                name_left_x + ranked_name.get_width() / 2,
                row_y,
                0.0,
            ]))

            if probability_value is None:
                ranked_bar_group = Tex(r"\vdots", font_size=22, color=GREY_A)
                ranked_bar_group.move_to(np.array([
                    bar_left_x_target + ranked_max_bar_width / 2,
                    row_y,
                    0.0,
                ]))
                ranked_value = Tex(r"\vdots", font_size=21, color=TEAL_B)
            else:
                ranked_track = RoundedRectangle(
                    width=ranked_max_bar_width,
                    height=0.13,
                    corner_radius=0.045,
                    stroke_color="#24464d",
                    stroke_width=0.55,
                    fill_color="#071419",
                    fill_opacity=0.84,
                )
                ranked_track.move_to(np.array([
                    bar_left_x_target + ranked_max_bar_width / 2,
                    row_y,
                    0.0,
                ]))
                ranked_width = ranked_max_bar_width * probability_value / 0.1
                ranked_fill = RoundedRectangle(
                    width=ranked_width,
                    height=0.13,
                    corner_radius=0.045,
                    stroke_color=TEAL_B,
                    stroke_width=0.35,
                    fill_color=TEAL_B,
                    fill_opacity=0.76,
                )
                ranked_fill.move_to(np.array([
                    bar_left_x_target + ranked_width / 2,
                    row_y,
                    0.0,
                ]))
                ranked_bar_group = VGroup(ranked_track, ranked_fill)
                ranked_value = Tex(probability_text, font_size=21, color=TEAL_B)
            ranked_value.move_to(np.array([
                value_left_x + ranked_value.get_width() / 2,
                row_y,
                0.0,
            ]))
            ranked_rows.add(VGroup(ranked_name, ranked_bar_group, ranked_value))

        ranked_probability_title = probability_title.copy()
        ranked_probability_title.next_to(ranked_rows, UP, buff=0.32)
        ranked_probability_group = VGroup(ranked_probability_title, ranked_rows)
        ranked_probability_group.move_to(softmax_frame.get_center() + DOWN * 0.05)

        probability_scene_group = VGroup(
            probability_output_group,
            prob_bar_tracks,
            prob_bar_fills,
        )
        self.play(
            FadeOut(softmax_operator_label, shift=LEFT * 0.05),
            FadeOut(softmax_formula, shift=LEFT * 0.05),
            FadeOut(softmax_input_group, shift=LEFT * 0.08),
            Transform(probability_scene_group, ranked_probability_group),
            run_time=1.05,
        )
        self.wait(0.8)
