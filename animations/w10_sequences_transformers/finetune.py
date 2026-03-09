"""
W10 — Practical Fine-Tuning

  Phase 1: Pre-train vs fine-tune cost.
           Timeline: huge corpus → long pre-train → general model →
           small task dataset → fast fine-tune.

  Phase 2: Tokenization & input pipeline.
           Raw sentence → WordPiece tokens → IDs → [CLS]/[SEP] →
           embeddings + positional encoding.

  Phase 3: Fine-tuning strategies side by side.
           Feature extraction / Full fine-tune / Gradual unfreeze —
           training loss + val accuracy curves for each.

  Phase 4: Learning rate sensitivity.
           Too-high LR → catastrophic forgetting (val loss spikes).
           Sweet spot lr ≈ 2e-5 → smooth convergence.

Render:
  ../../env/bin/manim -pql finetune.py FineTuning
  ../../env/bin/manim -pqh finetune.py FineTuning
"""

from manim import *
import numpy as np

C_BG    = "#0d1117"
C_WHITE = "#ffffff"
C_DIM   = "#8b949e"
C_BLUE  = "#58a6ff"
C_AMBER = "#ffa657"
C_GREEN = "#3fb950"
C_RED   = "#f78166"
C_GOLD  = "#e3b341"
C_TEAL  = "#2dd4bf"
C_PURP  = "#bc8cff"
C_DEAD  = "#3d444d"


def _box(label, pos, color, w=1.60, h=0.48, font_size=12):
    rect = RoundedRectangle(
        width=w, height=h, corner_radius=0.07,
        fill_color=color, fill_opacity=0.16,
        stroke_color=color, stroke_width=1.6,
    ).move_to(pos)
    txt = Text(label, font_size=font_size, color=color).move_to(pos)
    return VGroup(rect, txt)


def _arrow(start, end, color=C_DIM, sw=1.4):
    return Arrow(start, end, buff=0.07, color=color,
                 stroke_width=sw, max_tip_length_to_length_ratio=0.22)


class FineTuning(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()
        self._phase4()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Pre-train vs fine-tune cost
    # ══════════════════════════════════════════════════════════════════════════

    def _phase1(self):
        title = (
            Text("Practical Fine-Tuning", font_size=22, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.28)
        )
        sub = (
            Text(
                "Transfer learning: pay the pre-training cost once, "
                "reuse for many tasks cheaply",
                font_size=13, color=C_DIM,
            )
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        # ── Timeline ──────────────────────────────────────────────────────────
        Y     = 0.10
        # (x, label, color, box_width)
        nodes = [
            (-5.2, "Huge corpus\n(web-scale)",           C_DIM,   1.55),
            (-2.2, "Pre-training\n(days / GPU-months)",  C_BLUE,  2.00),
            ( 0.8, "General\nmodel",                     C_TEAL,  1.55),
            ( 3.0, "Task dataset\n(hundreds of examples)", C_AMBER, 2.00),
            ( 5.8, "Fine-tuned\nmodel",                  C_GREEN, 1.55),
        ]

        node_mobs = []
        for x, lbl, col, w in nodes:
            m = _box(lbl, [x, Y, 0], col, w=w, h=0.58, font_size=10)
            node_mobs.append(m)

        arrs = []
        for i in range(len(nodes) - 1):
            x0 = nodes[i][0]   + nodes[i][3]   / 2 + 0.06
            x1 = nodes[i+1][0] - nodes[i+1][3] / 2 - 0.06
            arrs.append(_arrow([x0, Y, 0], [x1, Y, 0], color=C_DIM))

        self.play(
            LaggedStart(*[FadeIn(m) for m in node_mobs], lag_ratio=0.15),
            run_time=0.65,
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrs], lag_ratio=0.15),
            run_time=0.55,
        )

        # Cost annotations
        cost_pre = (
            Text("millions of GPU-hours", font_size=10, color=C_RED)
            .next_to(arrs[1], UP, buff=0.62)
        )
        cost_ft = (
            Text("minutes on a laptop", font_size=10, color=C_GREEN)
            .next_to(arrs[3], UP, buff=0.62)
        )
        self.play(FadeIn(cost_pre), FadeIn(cost_ft), run_time=0.30)

        caption = (
            Text(
                "BERT, GPT, ViT — pre-trained once, fine-tuned thousands of times",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.8)

        self._title  = title
        self._p1_all = VGroup(sub, *node_mobs, *arrs, cost_pre, cost_ft, caption)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Tokenization & input pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = (
            Text(
                "Input pipeline — raw text → WordPiece tokens → "
                "IDs → embeddings + positional encoding",
                font_size=13, color=C_DIM,
            )
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        RAW = "great film !"
        tokens = ["[CLS]", "great", "film", "!", "[SEP]"]
        ids    = [101,      2307,    2143,  999,  102]
        N      = len(tokens)

        Y_RAW  =  1.40
        Y_TOK  =  0.55
        Y_ID   = -0.30
        Y_EMB  = -1.10

        tok_xs = np.linspace(-3.8, 3.8, N)
        tok_cols = [C_GOLD, C_BLUE, C_BLUE, C_BLUE, C_GOLD]

        # Raw sentence
        raw_box = _box(f'"{RAW}"', [0.0, Y_RAW, 0], C_DIM, w=2.20, h=0.44, font_size=13)
        self.play(FadeIn(raw_box), run_time=0.25)

        # Arrow down + label
        tok_arrow = _arrow([0.0, Y_RAW - 0.23, 0], [0.0, Y_TOK + 0.26, 0], color=C_DIM)
        tok_step  = Text("WordPiece tokeniser", font_size=10, color=C_DIM
                         ).next_to(tok_arrow, RIGHT, buff=0.10)
        self.play(GrowArrow(tok_arrow), FadeIn(tok_step), run_time=0.28)

        # Token boxes
        tok_boxes = [
            _box(tokens[i], [tok_xs[i], Y_TOK, 0], tok_cols[i], w=0.80, h=0.40, font_size=11)
            for i in range(N)
        ]
        self.play(
            LaggedStart(*[FadeIn(t) for t in tok_boxes], lag_ratio=0.12),
            run_time=0.50,
        )

        # ID row
        id_arrow = _arrow([0.0, Y_TOK - 0.21, 0], [0.0, Y_ID + 0.22, 0], color=C_DIM)
        id_lbl   = Text("vocabulary lookup", font_size=10, color=C_DIM
                        ).next_to(id_arrow, RIGHT, buff=0.10)
        self.play(GrowArrow(id_arrow), FadeIn(id_lbl), run_time=0.25)

        id_boxes = [
            _box(str(ids[i]), [tok_xs[i], Y_ID, 0], C_DIM, w=0.72, h=0.36, font_size=10)
            for i in range(N)
        ]
        self.play(
            LaggedStart(*[FadeIn(b) for b in id_boxes], lag_ratio=0.10),
            run_time=0.40,
        )

        # Embedding row
        emb_arrow = _arrow([0.0, Y_ID - 0.19, 0], [0.0, Y_EMB + 0.24, 0], color=C_DIM)
        emb_lbl   = Text("embedding table  +  PE", font_size=10, color=C_DIM
                         ).next_to(emb_arrow, RIGHT, buff=0.10)
        self.play(GrowArrow(emb_arrow), FadeIn(emb_lbl), run_time=0.25)

        CELL = 0.14
        D    = 8
        emb_grids = []
        for i in range(N):
            np.random.seed(ids[i] % 50)
            vals = np.random.randn(D)
            row  = VGroup()
            for d in range(D):
                v   = float(np.tanh(vals[d]))
                col = interpolate_color(ManimColor(C_PURP), ManimColor(C_TEAL),
                                        (v + 1) / 2)
                sq  = Square(side_length=CELL,
                             fill_color=col, fill_opacity=0.82,
                             stroke_color=C_DIM, stroke_width=0.3
                             ).move_to([tok_xs[i] + (d - D / 2 + 0.5) * CELL,
                                        Y_EMB, 0])
                row.add(sq)
            emb_grids.append(row)

        self.play(
            LaggedStart(*[FadeIn(g) for g in emb_grids], lag_ratio=0.10),
            run_time=0.50,
        )

        caption = (
            Text(
                "[CLS] and [SEP] are special tokens added by the tokeniser  ·  "
                "[CLS] output is used for classification",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.8)

        self._p2_all = VGroup(
            sub, raw_box, tok_arrow, tok_step, *tok_boxes,
            id_arrow, id_lbl, *id_boxes,
            emb_arrow, emb_lbl, *emb_grids, caption,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Fine-tuning strategies
    # ══════════════════════════════════════════════════════════════════════════

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.40)

        sub = (
            Text(
                "Fine-tuning strategies — how much of the backbone to update",
                font_size=13, color=C_DIM,
            )
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        epochs = np.arange(1, 21)

        def _exp(base, decay, noise, seed, offset=0.0):
            rng = np.random.default_rng(seed)
            return np.clip(base * np.exp(-decay * epochs) + noise * rng.random(20) + offset,
                           0.05, 2.0)

        strategies = [
            {
                "title":  "Feature Extraction",
                "color":  C_BLUE,
                "cx":     -4.0,
                "train":  _exp(1.6, 0.08, 0.05, 1, 0.25),
                "val":    _exp(1.5, 0.07, 0.06, 2, 0.30),
                "note":   "backbone frozen\nonly head trains",
            },
            {
                "title":  "Full Fine-Tune",
                "color":  C_AMBER,
                "cx":      0.0,
                "train":  _exp(1.6, 0.13, 0.04, 3),
                "val":    _exp(1.5, 0.09, 0.06, 4, 0.18),
                "note":   "all layers update\nhigh LR risk",
            },
            {
                "title":  "Gradual Unfreeze",
                "color":  C_GREEN,
                "cx":      4.0,
                "train":  _exp(1.6, 0.12, 0.04, 5, 0.05),
                "val":    _exp(1.5, 0.11, 0.04, 6, 0.06),
                "note":   "unfreeze layer by layer\nbest for small data",
            },
        ]

        AX_W = 2.80
        AX_H = 1.80
        Y_TOP = 0.70
        all_mobs = VGroup()

        for s in strategies:
            cx = s["cx"]
            ax = Axes(
                x_range=[0, 21, 5], y_range=[0, 2.2, 0.5],
                x_length=AX_W, y_length=AX_H,
                axis_config={"color": C_DIM, "stroke_width": 1.0,
                             "include_ticks": False},
            ).move_to([cx, Y_TOP - AX_H / 2, 0])

            hdr = Text(s["title"], font_size=12, color=s["color"], weight=BOLD
                       ).next_to(ax, UP, buff=0.12)
            x_lbl = Text("Epochs", font_size=9, color=C_DIM).next_to(ax, DOWN, buff=0.06)
            y_lbl = (Text("Loss", font_size=9, color=C_DIM)
                     .rotate(PI / 2).next_to(ax, LEFT, buff=0.06))

            self.play(Create(ax), FadeIn(hdr), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.35)

            def _pts(arr):
                return [ax.c2p(e, arr[i]) for i, e in enumerate(epochs)]

            t_path = VMobject(stroke_color=C_BLUE,  stroke_width=1.8)
            v_path = VMobject(stroke_color=C_RED,   stroke_width=1.8)
            t_path.set_points_as_corners(_pts(s["train"]))
            v_path.set_points_as_corners(_pts(s["val"]))

            self.play(Create(t_path), Create(v_path), run_time=0.60)

            # Legend
            leg = VGroup(
                VGroup(Line([0,0,0],[0.25,0,0], stroke_color=C_BLUE, stroke_width=1.8),
                       Text("train", font_size=9, color=C_BLUE)).arrange(RIGHT, buff=0.07),
                VGroup(Line([0,0,0],[0.25,0,0], stroke_color=C_RED,  stroke_width=1.8),
                       Text("val",   font_size=9, color=C_RED)).arrange(RIGHT, buff=0.07),
            ).arrange(RIGHT, buff=0.20).next_to(ax, DOWN, buff=0.20)

            note = Text(s["note"], font_size=9, color=s["color"]
                        ).next_to(leg, DOWN, buff=0.12)

            self.play(FadeIn(leg), FadeIn(note), run_time=0.25)
            all_mobs.add(ax, hdr, x_lbl, y_lbl, t_path, v_path, leg, note)

        caption = (
            Text(
                "gradual unfreeze maintains pre-trained representations "
                "while adapting to the new task",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(2.0)

        self._p3_all = VGroup(sub, all_mobs, caption)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4 — Learning rate sensitivity
    # ══════════════════════════════════════════════════════════════════════════

    def _phase4(self):
        self.play(FadeOut(self._p3_all), run_time=0.40)

        sub = (
            Text(
                "Learning rate sensitivity — too high destroys "
                "pre-trained weights (catastrophic forgetting)",
                font_size=13, color=C_DIM,
            )
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        epochs = np.arange(1, 31)

        def _loss(base, decay, noise, seed, offset=0.0):
            rng = np.random.default_rng(seed)
            return np.clip(base * np.exp(-decay * epochs) + noise * rng.random(30) + offset,
                           0.04, 2.5)

        # Too high LR — val loss spikes after a few epochs
        rng = np.random.default_rng(20)
        val_high = np.concatenate([
            _loss(1.5, 0.18, 0.04, 10)[:8],
            np.clip(1.2 + 0.06 * np.arange(22) + 0.05 * rng.random(22), 0.8, 2.5),
        ])
        train_high = _loss(1.6, 0.15, 0.04, 11)

        # Sweet spot lr ≈ 2e-5 — smooth convergence
        train_sweet = _loss(1.6, 0.11, 0.04, 12, 0.04)
        val_sweet   = _loss(1.5, 0.10, 0.04, 13, 0.05)

        # Too low LR — barely moves
        train_low = _loss(1.6, 0.03, 0.03, 14, 0.30)
        val_low   = _loss(1.5, 0.025, 0.03, 15, 0.35)

        lr_specs = [
            ("lr = 1e-3  (too high)",  C_RED,   -4.0, train_high, val_high),
            ("lr = 2e-5  (sweet spot)", C_GREEN,  0.0, train_sweet, val_sweet),
            ("lr = 1e-7  (too low)",   C_AMBER,   4.0, train_low,  val_low),
        ]

        AX_W = 2.80
        AX_H = 1.80
        Y_TOP = 0.70
        all_mobs = VGroup()

        for lbl, col, cx, t_loss, v_loss in lr_specs:
            ax = Axes(
                x_range=[0, 31, 10], y_range=[0, 2.5, 0.5],
                x_length=AX_W, y_length=AX_H,
                axis_config={"color": C_DIM, "stroke_width": 1.0,
                             "include_ticks": False},
            ).move_to([cx, Y_TOP - AX_H / 2, 0])

            hdr   = Text(lbl, font_size=11, color=col, weight=BOLD
                         ).next_to(ax, UP, buff=0.12)
            x_lbl = Text("Epochs", font_size=9, color=C_DIM).next_to(ax, DOWN, buff=0.06)
            y_lbl = (Text("Val Loss", font_size=9, color=C_DIM)
                     .rotate(PI / 2).next_to(ax, LEFT, buff=0.06))

            self.play(Create(ax), FadeIn(hdr), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.32)

            def _pts(arr):
                return [ax.c2p(e, arr[i]) for i, e in enumerate(epochs)]

            t_path = VMobject(stroke_color=C_BLUE,  stroke_width=1.8)
            v_path = VMobject(stroke_color=col,     stroke_width=1.8)
            t_path.set_points_as_corners(_pts(t_loss))
            v_path.set_points_as_corners(_pts(v_loss))

            self.play(Create(t_path), Create(v_path), run_time=0.65)

            leg = VGroup(
                VGroup(Line([0,0,0],[0.25,0,0], stroke_color=C_BLUE, stroke_width=1.8),
                       Text("train", font_size=9, color=C_BLUE)).arrange(RIGHT, buff=0.07),
                VGroup(Line([0,0,0],[0.25,0,0], stroke_color=col, stroke_width=1.8),
                       Text("val",   font_size=9, color=col)).arrange(RIGHT, buff=0.07),
            ).arrange(RIGHT, buff=0.20).next_to(ax, DOWN, buff=0.20)
            self.play(FadeIn(leg), run_time=0.22)
            all_mobs.add(ax, hdr, x_lbl, y_lbl, t_path, v_path, leg)

        # Annotation on high-LR panel
        forget_note = (
            Text("catastrophic\nforgetting", font_size=10, color=C_RED)
            .move_to([-4.0, -1.80, 0])
        )
        self.play(FadeIn(forget_note), run_time=0.25)

        caption = (
            Text(
                "rule of thumb: lr ≈ 2e-5 for BERT  ·  "
                "use a scheduler (linear warmup + decay) for best results",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(2.5)
