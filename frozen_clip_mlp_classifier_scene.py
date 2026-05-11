from manimlib import *


class FrozenCLIPMLPClassifierScene(Scene):
    def make_face_crop(self, side=1.45):
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

    def make_feature_vector(self, color=YELLOW_B, font_size=36):
        vector = Tex(
            r"\left[\begin{array}{c}"
            r"n_1\\"
            r"n_2\\"
            r"\vdots\\"
            r"n_{768}"
            r"\end{array}\right]",
            font_size=font_size - 4,
            color=color,
        )
        label = Tex(r"768\text{-dim face vector}", font_size=22, color=GREY_A)
        label.next_to(vector, DOWN, buff=0.16)
        return VGroup(vector, label)

    def make_encoder_block(self):
        box = RoundedRectangle(
            width=2.05,
            height=1.02,
            corner_radius=0.1,
            stroke_color=BLUE_B,
            stroke_width=1.35,
            fill_color="#111820",
            fill_opacity=0.92,
        )
        title = TexText("Image Encoder", font_size=22, color=WHITE)
        detail = TexText("frozen", font_size=17, color=YELLOW_B)
        detail.next_to(title, DOWN, buff=0.08)
        content = VGroup(title, detail)
        content.move_to(box)
        return VGroup(box, content)

    def make_arrow_between(self, left, right, color=GREY_A, buff=0.16):
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

    def make_name_logits(self):
        names = ["David", "Daniel", "Mary", "Other"]
        values = [1.72, 0.95, 0.52, 0.28]
        colors = [YELLOW_B, BLUE_B, TEAL_B, GREY_B]
        bars = VGroup()
        labels = VGroup()
        scores = VGroup()

        for name, value, color in zip(names, values, colors):
            bar = Rectangle(width=value, height=0.28)
            bar.set_fill(color, opacity=0.62)
            bar.set_stroke(color, width=1.2, opacity=0.96)
            bars.add(bar)

            label = TexText(name, font_size=18, color=WHITE)
            labels.add(label)

            score = Tex(f"{value:.2f}", font_size=18, color=color)
            scores.add(score)

        bars.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        for label, bar, score in zip(labels, bars, scores):
            label.next_to(bar, LEFT, buff=0.18)
            score.next_to(bar, RIGHT, buff=0.12)

        title = TexText("predicted name", font_size=21, color=YELLOW_B)
        title.next_to(bars, UP, buff=0.18)
        group = VGroup(title, labels, bars, scores)
        return group

    def construct(self):
        self.camera.background_color = BLACK

        clip_frame = RoundedRectangle(
            width=11.7,
            height=6.15,
            corner_radius=0.14,
            stroke_color=BLUE_B,
            stroke_width=1.2,
            fill_color=GREY_E,
            fill_opacity=0.035,
        )
        clip_frame.move_to(DOWN * 0.04)
        clip_title = Tex(r"\text{Frozen CLIP image encoder}", font_size=36, color=WHITE)
        clip_title.move_to(clip_frame.get_top() + DOWN * 0.27)
        clip_subtitle = Tex(
            r"\text{weights frozen: representation only}",
            font_size=22,
            color=GREY_A,
        )
        clip_subtitle.next_to(clip_frame, DOWN, buff=0.12)

        face = self.make_face_crop(side=1.48)
        face[1].move_to(LEFT * 4.1)
        face[0].move_to(face[1])
        face[2].next_to(face[1], DOWN, buff=0.12)
        encoder = self.make_encoder_block()
        encoder.move_to(LEFT * 0.85)
        feature_vector = self.make_feature_vector()
        feature_vector[0].move_to(RIGHT * 3.35)
        feature_vector[1].next_to(feature_vector[0], DOWN, buff=0.16)

        face_to_encoder = self.make_arrow_between(face[1], encoder, color=BLUE_B)
        encoder_to_vector = self.make_arrow_between(encoder, feature_vector[0], color=YELLOW_B)

        self.play(
            FadeIn(clip_frame),
            FadeIn(clip_title, shift=UP * 0.06),
            FadeIn(clip_subtitle, shift=UP * 0.04),
            FadeIn(face, shift=UP * 0.08),
            FadeIn(encoder, shift=UP * 0.08),
            FadeIn(feature_vector, shift=UP * 0.08),
            run_time=0.9,
        )
        self.play(
            encoder.animate.scale(1.08),
            clip_frame.animate.set_stroke(YELLOW_B, width=1.4, opacity=1.0),
            run_time=0.34,
        )
        self.play(
            encoder.animate.scale(1 / 1.08),
            clip_frame.animate.set_stroke(BLUE_B, width=1.2, opacity=0.95),
            run_time=0.34,
        )
        self.play(
            ShowCreation(face_to_encoder),
            ShowCreation(encoder_to_vector),
            run_time=0.75,
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
        mlp_frame.move_to(clip_frame)
        mlp_title = Tex(r"\text{Trained MLP classifier}", font_size=36, color=WHITE)
        mlp_title.move_to(mlp_frame.get_top() + DOWN * 0.27)
        mlp_subtitle = Tex(
            r"\text{only this head learns face}\rightarrow\text{name}",
            font_size=22,
            color=GREY_A,
        )
        mlp_subtitle.next_to(mlp_frame, DOWN, buff=0.12)

        vector_source = feature_vector.copy()
        vector_source.scale(0.66)
        vector_source[0].move_to(LEFT * 4.0)
        vector_source[1].next_to(vector_source[0], DOWN, buff=0.14)
        mlp = self.make_mlp_network()
        mlp.move_to(LEFT * 0.45)
        logits = self.make_name_logits()
        logits.move_to(RIGHT * 3.65)
        source_to_mlp = self.make_arrow_between(vector_source[0], mlp, color=GREEN_B)
        mlp_to_logits = self.make_arrow_between(mlp, logits, color=GREEN_B)

        trainable_tag = TexText("trainable", font_size=21, color=GREEN_B)
        trainable_tag.next_to(mlp, DOWN, buff=0.48)

        self.play(
            FadeOut(face_to_encoder),
            FadeOut(encoder_to_vector),
            FadeOut(face, shift=LEFT * 0.10),
            FadeOut(encoder, shift=DOWN * 0.08),
            Transform(clip_frame, mlp_frame),
            Transform(clip_title, mlp_title),
            Transform(clip_subtitle, mlp_subtitle),
            Transform(feature_vector, vector_source),
            run_time=1.0,
        )
        self.play(
            ShowCreation(source_to_mlp),
            FadeIn(mlp, shift=RIGHT * 0.08),
            FadeIn(trainable_tag, shift=UP * 0.05),
            run_time=0.85,
        )
        self.play(
            mlp[0].animate.set_stroke(GREEN_B, width=1.25, opacity=0.76),
            mlp[1][0].animate.set_fill(YELLOW_B, opacity=0.42),
            run_time=0.45,
        )
        self.play(
            mlp[1][0].animate.set_fill(YELLOW_B, opacity=0.10),
            mlp[1][1].animate.set_fill(GREEN_B, opacity=0.46),
            run_time=0.45,
        )
        self.play(
            mlp[1][1].animate.set_fill(GREEN_B, opacity=0.10),
            mlp[1][2].animate.set_fill(BLUE_B, opacity=0.46),
            ShowCreation(mlp_to_logits),
            FadeIn(logits, shift=RIGHT * 0.10),
            run_time=0.75,
        )
        self.play(
            logits[2][0].animate.set_fill(YELLOW_B, opacity=0.88).set_stroke(YELLOW_B, width=2.4, opacity=1.0),
            logits[1][0].animate.set_color(YELLOW_B),
            run_time=0.45,
        )
        self.wait(1.2)
