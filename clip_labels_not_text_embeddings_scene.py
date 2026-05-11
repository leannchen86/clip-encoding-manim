from manimlib import *


class CLIPLabelsNotTextEmbeddingsScene(Scene):
    def make_face_crop(self, side=1.18, label_text="face crop"):
        image = ImageMobject("croped1.png")
        image.set_height(side)
        if image.get_width() > side:
            image.set_width(side)

        frame = Square(side_length=side + 0.05)
        frame.set_fill("#111318", opacity=0.18)
        frame.set_stroke(WHITE, width=1.45, opacity=0.88)
        frame.move_to(image)

        label = TexText(label_text, font_size=17, color=GREY_A)
        label.next_to(frame, DOWN, buff=0.11)
        return Group(image, frame, label)

    def make_encoder_block(self, title_text, detail_text, color=BLUE_B, width=2.05):
        box = RoundedRectangle(
            width=width,
            height=1.02,
            corner_radius=0.1,
            stroke_color=color,
            stroke_width=1.35,
            fill_color="#111820",
            fill_opacity=0.92,
        )
        title = TexText(title_text, font_size=21, color=WHITE)
        detail = TexText(detail_text, font_size=16, color=YELLOW_B if color == BLUE_B else GREY_A)
        detail.next_to(title, DOWN, buff=0.08)
        content = VGroup(title, detail)
        content.move_to(box)
        return VGroup(box, content)

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

    def make_feature_vector(self, color=YELLOW_B, font_size=31):
        vector = Tex(
            r"\left[\begin{array}{c}"
            r"n_1\\"
            r"n_2\\"
            r"\vdots\\"
            r"n_{768}"
            r"\end{array}\right]",
            font_size=font_size,
            color=color,
        )
        label = Tex(r"768\text{-dim face vector}", font_size=20, color=GREY_A)
        label.next_to(vector, DOWN, buff=0.14)
        return VGroup(vector, label)

    def make_caption_card(self, text, width=2.38):
        card = RoundedRectangle(
            width=width,
            height=0.62,
            corner_radius=0.08,
            stroke_color=GREY_B,
            stroke_width=1.05,
            fill_color="#101217",
            fill_opacity=0.92,
        )
        label = TexText(text, font_size=17, color=WHITE)
        label.move_to(card)
        return VGroup(card, label)

    def make_shared_space(self):
        axes = VGroup(
            Line(LEFT * 1.06, RIGHT * 1.06),
            Line(DOWN * 0.72, UP * 0.72),
        )
        axes.set_stroke(GREY_B, width=0.9, opacity=0.42)

        dots = VGroup()
        specs = [
            (LEFT * 0.42 + UP * 0.24, BLUE_B),
            (LEFT * 0.16 + DOWN * 0.15, BLUE_B),
            (RIGHT * 0.34 + UP * 0.12, YELLOW_B),
            (RIGHT * 0.48 + DOWN * 0.22, YELLOW_B),
            (LEFT * 0.62 + DOWN * 0.33, TEAL_B),
            (RIGHT * 0.08 + UP * 0.42, TEAL_B),
        ]
        for point, color in specs:
            dot = Dot(point, radius=0.045)
            dot.set_color(color)
            dot.set_opacity(0.85)
            dots.add(dot)

        halo = Ellipse(width=2.15, height=1.48)
        halo.set_fill(BLUE_B, opacity=0.04)
        halo.set_stroke(WHITE, width=1.05, opacity=0.58)

        label = TexText("shared CLIP space", font_size=18, color=WHITE)
        label.next_to(halo, DOWN, buff=0.12)
        return VGroup(halo, axes, dots, label)

    def make_name_tokens(self):
        tokens = VGroup()
        for text in ['"David"', '"John"']:
            token = RoundedRectangle(
                width=1.12,
                height=0.36,
                corner_radius=0.07,
                stroke_color=GREY_B,
                stroke_width=1.05,
                fill_color="#14161c",
                fill_opacity=0.92,
            )
            label = TexText(text, font_size=16, color=WHITE)
            label.move_to(token)
            tokens.add(VGroup(token, label))
        tokens.arrange(DOWN, buff=0.16)
        title = TexText("names as text", font_size=16, color=GREY_A)
        title.next_to(tokens, UP, buff=0.12)
        return VGroup(title, tokens)

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

    def make_output_mapping(self):
        rows = VGroup()
        entries = [
            ("0", "David", YELLOW_B),
            ("1", "John", BLUE_B),
            ("2", "Mary", TEAL_B),
            (r"\cdots", "...", GREY_A),
        ]
        for index, name, color in entries:
            row_box = RoundedRectangle(
                width=2.34,
                height=0.36,
                corner_radius=0.06,
                stroke_color=color,
                stroke_width=1.05,
                fill_color="#111820",
                fill_opacity=0.88,
            )
            index_label = Tex(index, font_size=18, color=color)
            name_label = TexText(name, font_size=17, color=WHITE)
            index_label.move_to(row_box.get_left() + RIGHT * 0.34)
            name_label.move_to(row_box.get_center() + RIGHT * 0.34)
            rows.add(VGroup(row_box, index_label, name_label))
        rows.arrange(DOWN, buff=0.1)

        title = TexText("output index -> class label", font_size=19, color=GREEN_B)
        title.next_to(rows, UP, buff=0.16)

        checkpoint = RoundedRectangle(
            width=2.05,
            height=0.46,
            corner_radius=0.07,
            stroke_color=GREEN_B,
            stroke_width=1.05,
            fill_color="#101217",
            fill_opacity=0.92,
        )
        checkpoint_label = TexText("saved in checkpoint", font_size=16, color=GREY_A)
        checkpoint_label.move_to(checkpoint)
        checkpoint_group = VGroup(checkpoint, checkpoint_label)
        checkpoint_group.next_to(rows, DOWN, buff=0.16)

        return VGroup(title, rows, checkpoint_group)

    def construct(self):
        self.camera.background_color = BLACK

        intro_encoder = self.make_encoder_block("Image Encoder", "frozen", color=BLUE_B)
        intro_encoder.move_to(ORIGIN)

        pretrain_frame = RoundedRectangle(
            width=11.7,
            height=6.15,
            corner_radius=0.14,
            stroke_color=BLUE_B,
            stroke_width=1.2,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        pretrain_frame.move_to(DOWN * 0.04)
        pretrain_title = Tex(r"\text{CLIP pretraining built the embedding space}", font_size=34, color=WHITE)
        pretrain_title.move_to(pretrain_frame.get_top() + DOWN * 0.27)
        pretrain_subtitle = Tex(
            r"\text{image encoder and text encoder were trained together on image-caption pairs}",
            font_size=20,
            color=GREY_A,
        )
        pretrain_subtitle.next_to(pretrain_frame, DOWN, buff=0.12)

        image_encoder = self.make_encoder_block("Image Encoder", "pretraining", color=BLUE_B)
        image_encoder.move_to(LEFT * 2.35 + UP * 0.72)
        text_encoder = self.make_encoder_block("Text Encoder", "pretraining", color=TEAL_B)
        text_encoder.move_to(LEFT * 2.35 + DOWN * 0.74)

        shared_space = self.make_shared_space()
        shared_space.move_to(RIGHT * 2.55)

        image_to_space = self.make_arrow_between(image_encoder, shared_space[0], color=BLUE_B)
        text_to_space = self.make_arrow_between(text_encoder, shared_space[0], color=TEAL_B)

        self.play(
            FadeIn(intro_encoder, shift=UP * 0.08),
            run_time=0.7,
        )
        self.play(
            intro_encoder.animate.scale(1.08),
            run_time=0.34,
        )
        self.play(
            intro_encoder.animate.scale(1 / 1.08),
            run_time=0.34,
        )
        self.play(
            ReplacementTransform(intro_encoder[0], pretrain_frame),
            FadeOut(intro_encoder[1]),
            FadeIn(pretrain_title, shift=UP * 0.06),
            FadeIn(pretrain_subtitle, shift=UP * 0.04),
            run_time=0.95,
        )
        self.play(
            FadeIn(image_encoder, shift=RIGHT * 0.08),
            FadeIn(text_encoder, shift=RIGHT * 0.08),
            run_time=0.75,
        )
        self.play(
            ShowCreation(image_to_space),
            ShowCreation(text_to_space),
            FadeIn(shared_space, shift=RIGHT * 0.08),
            run_time=0.85,
        )
        self.play(
            shared_space[0].animate.set_stroke(YELLOW_B, width=1.7, opacity=0.95),
            run_time=0.45,
        )
        self.play(
            shared_space[0].animate.set_stroke(WHITE, width=1.05, opacity=0.58),
            run_time=0.35,
        )
        self.wait(0.65)

        pipeline_frame = RoundedRectangle(
            width=11.7,
            height=6.15,
            corner_radius=0.14,
            stroke_color=YELLOW_B,
            stroke_width=1.25,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        pipeline_frame.move_to(pretrain_frame)
        pipeline_title = Tex(r"\text{Our pipeline uses only CLIP's image encoder}", font_size=34, color=WHITE)
        pipeline_title.move_to(pipeline_frame.get_top() + DOWN * 0.27)
        pipeline_subtitle = Tex(
            r"\text{we do not run the text encoder or embed names with CLIP}",
            font_size=21,
            color=GREY_A,
        )
        pipeline_subtitle.next_to(pipeline_frame, DOWN, buff=0.12)

        face = self.make_face_crop(side=1.25, label_text="face crop")
        face[1].move_to(LEFT * 4.15)
        face[0].move_to(face[1])
        face[2].next_to(face[1], DOWN, buff=0.11)
        frozen_encoder = self.make_encoder_block("Image Encoder", "frozen", color=BLUE_B)
        frozen_encoder.move_to(LEFT * 1.05)
        vector = self.make_feature_vector()
        vector[0].move_to(RIGHT * 2.05)
        vector[1].next_to(vector[0], DOWN, buff=0.14)

        name_tokens = self.make_name_tokens()
        name_tokens.move_to(RIGHT * 4.75 + UP * 0.38)
        disabled_text_encoder = self.make_encoder_block("Text Encoder", "not used", color=GREY_B, width=1.82)
        disabled_text_encoder.move_to(RIGHT * 4.75 + DOWN * 0.88)

        face_to_frozen = self.make_arrow_between(face[1], frozen_encoder, color=BLUE_B)
        frozen_to_vector = self.make_arrow_between(frozen_encoder, vector[0], color=YELLOW_B)

        self.play(
            FadeOut(VGroup(image_to_space, text_to_space)),
            FadeOut(VGroup(image_encoder, shared_space)),
            Transform(pretrain_frame, pipeline_frame),
            Transform(pretrain_title, pipeline_title),
            Transform(pretrain_subtitle, pipeline_subtitle),
            Transform(text_encoder, disabled_text_encoder),
            run_time=1.0,
        )
        self.play(
            FadeIn(face, shift=UP * 0.08),
            FadeIn(frozen_encoder, shift=UP * 0.08),
            FadeIn(vector, shift=UP * 0.08),
            FadeIn(name_tokens, shift=UP * 0.06),
            run_time=0.85,
        )
        self.play(
            frozen_encoder.animate.scale(1.08),
            pretrain_frame.animate.set_stroke(BLUE_B, width=1.4, opacity=1.0),
            run_time=0.34,
        )
        self.play(
            frozen_encoder.animate.scale(1 / 1.08),
            pretrain_frame.animate.set_stroke(YELLOW_B, width=1.25, opacity=0.95),
            run_time=0.34,
        )
        self.play(
            ShowCreation(face_to_frozen),
            ShowCreation(frozen_to_vector),
            run_time=0.85,
        )
        self.play(
            text_encoder[0].animate.set_stroke(RED_B, width=2.1, opacity=1.0).set_fill("#321416", opacity=0.96),
            text_encoder[1].animate.set_color(RED_B),
            name_tokens[0].animate.set_color(RED_B),
            *[
                token[0].animate.set_stroke(RED_B, width=1.75, opacity=1.0).set_fill("#321416", opacity=0.96)
                for token in name_tokens[1]
            ],
            *[
                token[1].animate.set_color(RED_B)
                for token in name_tokens[1]
            ],
            run_time=0.32,
        )
        self.play(
            text_encoder.animate.scale(1.08),
            name_tokens.animate.scale(1.08),
            run_time=0.22,
        )
        self.play(
            text_encoder.animate.scale(1 / 1.08),
            name_tokens.animate.scale(1 / 1.08),
            run_time=0.22,
        )
        self.wait(0.28)
        self.play(
            text_encoder.animate.set_opacity(0.34),
            name_tokens.animate.set_opacity(0.34),
            run_time=0.38,
        )
        self.wait(0.75)

        mlp_frame = RoundedRectangle(
            width=11.7,
            height=6.15,
            corner_radius=0.14,
            stroke_color=GREEN_B,
            stroke_width=1.25,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        mlp_frame.move_to(pretrain_frame)
        mlp_title = Tex(r"\text{Names are class labels, not CLIP text embeddings}", font_size=34, color=WHITE)
        mlp_title.move_to(mlp_frame.get_top() + DOWN * 0.27)
        mlp_subtitle = Tex(
            r"\text{the checkpoint saves which output position means which name}",
            font_size=21,
            color=GREY_A,
        )
        mlp_subtitle.next_to(mlp_frame, DOWN, buff=0.12)

        vector_source = vector.copy()
        vector_source.scale(0.66)
        vector_source[0].move_to(LEFT * 4.0)
        vector_source[1].next_to(vector_source[0], DOWN, buff=0.14)
        mlp = self.make_mlp_network()
        mlp.move_to(LEFT * 0.45)
        mapping = self.make_output_mapping()
        mapping.move_to(RIGHT * 3.65)
        vector_to_mlp = self.make_arrow_between(vector_source[0], mlp, color=GREEN_B)
        mlp_to_mapping = self.make_arrow_between(mlp, mapping, color=GREEN_B)

        class_label_tag = TexText("class labels only", font_size=21, color=GREEN_B)
        class_label_tag.next_to(mapping, DOWN, buff=0.34)

        self.play(
            FadeOut(face_to_frozen),
            FadeOut(frozen_to_vector),
            FadeOut(face, shift=LEFT * 0.10),
            FadeOut(frozen_encoder, shift=DOWN * 0.08),
            FadeOut(text_encoder, shift=DOWN * 0.08),
            FadeOut(name_tokens, shift=RIGHT * 0.08),
            Transform(pretrain_frame, mlp_frame),
            Transform(pretrain_title, mlp_title),
            Transform(pretrain_subtitle, mlp_subtitle),
            Transform(vector, vector_source),
            run_time=1.0,
        )
        self.play(
            ShowCreation(vector_to_mlp),
            FadeIn(mlp, shift=RIGHT * 0.08),
            run_time=0.75,
        )
        self.play(
            mlp[0].animate.set_stroke(GREEN_B, width=1.25, opacity=0.76),
            mlp[1][0].animate.set_fill(YELLOW_B, opacity=0.42),
            run_time=0.42,
        )
        self.play(
            mlp[1][0].animate.set_fill(YELLOW_B, opacity=0.10),
            mlp[1][1].animate.set_fill(GREEN_B, opacity=0.46),
            run_time=0.42,
        )
        self.play(
            mlp[1][1].animate.set_fill(GREEN_B, opacity=0.10),
            mlp[1][2].animate.set_fill(BLUE_B, opacity=0.46),
            ShowCreation(mlp_to_mapping),
            FadeIn(mapping, shift=RIGHT * 0.08),
            run_time=0.78,
        )
        self.play(
            mapping[1][0][0].animate.set_fill(YELLOW_B, opacity=0.20).set_stroke(YELLOW_B, width=1.85, opacity=1.0),
            FadeIn(class_label_tag, shift=UP * 0.05),
            run_time=0.55,
        )
        self.play(
            mapping[2][0].animate.set_stroke(GREEN_B, width=1.85, opacity=1.0),
            run_time=0.45,
        )
        self.wait(1.2)
