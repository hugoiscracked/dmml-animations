"""
W09 — CNN Architecture

  Phase 1: LeNet-style pipeline
           Input image → Conv1 → Pool1 → Conv2 → Pool2 → FC → Softmax
           Each stage labelled; feature-map volumes shown as 3-D cuboids.

  Phase 2: Hierarchical features
           Three columns (early / mid / late layer) with example
           feature descriptions: edges & colours → textures & shapes
           → object parts → class scores.

  Phase 3: Transfer learning — freeze / fine-tune
           Pre-trained backbone shown with frozen (blue, locked) layers
           and a small new head (amber, unlocked). Two-step reveal:
           first freeze + train head, then unfreeze top layers for
           fine-tuning.

Render:
  ../../env/bin/manim -pql cnn_architecture.py CNNArchitecture
  ../../env/bin/manim -pqh cnn_architecture.py CNNArchitecture
"""

from manim import *
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
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


# ── Cuboid helper ─────────────────────────────────────────────────────────────
def _cuboid(w, h, d, color, fill_op=0.18, stroke_w=1.4):
    """
    Return a VGroup that looks like an axis-aligned cuboid
    (front face + top face + right face).
    w = width, h = height, d = depth offset (isometric projection).
    Origin is the bottom-left of the front face.
    """
    dx = d * 0.40   # isometric x shift
    dy = d * 0.22   # isometric y shift

    # Front face (rectangle)
    front = Polygon(
        [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],
        fill_color=color, fill_opacity=fill_op,
        stroke_color=color, stroke_width=stroke_w,
    )
    # Top face
    top = Polygon(
        [0, h, 0], [w, h, 0], [w + dx, h + dy, 0], [dx, h + dy, 0],
        fill_color=color, fill_opacity=fill_op * 1.4,
        stroke_color=color, stroke_width=stroke_w,
    )
    # Right face
    right = Polygon(
        [w, 0, 0], [w + dx, dy, 0], [w + dx, h + dy, 0], [w, h, 0],
        fill_color=color, fill_opacity=fill_op * 0.7,
        stroke_color=color, stroke_width=stroke_w,
    )
    return VGroup(front, top, right)


def _cuboid_centred(w, h, d, color, **kwargs):
    """Return cuboid shifted so the front face is centred at origin."""
    cub = _cuboid(w, h, d, color, **kwargs)
    cub.shift([-w / 2, -h / 2, 0])
    return cub


class CNNArchitecture(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — LeNet-style pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def _phase1(self):
        title = (
            Text("CNN Architecture", font_size=22, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.28)
        )
        sub = (
            Text(
                "LeNet-style pipeline: convolutions extract features, "
                "pooling reduces size, FC layers classify",
                font_size=13, color=C_DIM,
            )
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        # ── Stage definitions ─────────────────────────────────────────────────
        # Each entry: (label, w, h, d, color, annotation)
        stages = [
            ("Input\n28×28",    0.60, 1.60, 0.10, C_WHITE,  None),
            ("Conv1\n6 maps",   0.22, 1.40, 0.55, C_BLUE,   "3×3 filters\nstride 1"),
            ("Pool1\n6 maps",   0.22, 0.90, 0.55, C_TEAL,   "2×2 max\n÷ 2 size"),
            ("Conv2\n16 maps",  0.22, 0.80, 0.90, C_PURP,   "3×3 filters\nstride 1"),
            ("Pool2\n16 maps",  0.22, 0.50, 0.90, C_TEAL,   "2×2 max\n÷ 2 size"),
            ("FC\n120",         0.28, 1.00, 0.14, C_AMBER,  "flatten\n+ ReLU"),
            ("FC\n84",          0.28, 0.70, 0.14, C_AMBER,  None),
            ("Softmax\n10",     0.28, 0.42, 0.14, C_GREEN,  "class\nscores"),
        ]

        n = len(stages)
        # Spread across x; leave room for annotations above/below
        xs = np.linspace(-5.6, 5.6, n)
        y_centre = -0.30    # vertical centre of all cuboids

        cuboids   = []
        lbl_mobs  = []
        ann_mobs  = []
        arrows    = []

        for i, (lbl, w, h, d, col, ann) in enumerate(stages):
            cub = _cuboid_centred(w, h, d, col).move_to([xs[i], y_centre, 0])
            cuboids.append(cub)

            # Label below
            lbl_mob = (
                Text(lbl, font_size=10, color=col)
                .next_to(cub, DOWN, buff=0.18)
            )
            lbl_mobs.append(lbl_mob)

            # Optional annotation above (alternating up/down to avoid clashes)
            if ann:
                side = UP if i % 2 == 1 else DOWN
                # For annotations we always put them above to keep it cleaner
                ann_mob = (
                    Text(ann, font_size=9, color=C_DIM)
                    .next_to(cub, UP, buff=0.22)
                )
                ann_mobs.append(ann_mob)

            # Arrow from previous stage
            if i > 0:
                prev = cuboids[i - 1]
                # right edge of prev front face ≈ prev centre x + w_prev/2
                x_start = xs[i - 1] + stages[i - 1][1] / 2
                x_end   = xs[i]     - w / 2
                y_arr   = y_centre
                arr = Arrow(
                    [x_start + 0.04, y_arr, 0],
                    [x_end   - 0.04, y_arr, 0],
                    buff=0.0, color=C_DIM, stroke_width=1.2,
                    max_tip_length_to_length_ratio=0.30,
                )
                arrows.append(arr)

        # ── Animate pipeline stage by stage ───────────────────────────────────
        # Show input
        self.play(FadeIn(cuboids[0]), FadeIn(lbl_mobs[0]), run_time=0.35)

        ann_idx = 0
        for i in range(1, n):
            lbl, w, h, d, col, ann = stages[i]
            anims = [
                GrowArrow(arrows[i - 1]),
                FadeIn(cuboids[i]),
                FadeIn(lbl_mobs[i]),
            ]
            if ann:
                anims.append(FadeIn(ann_mobs[ann_idx]))
                ann_idx += 1
            self.play(*anims, run_time=0.38)

        self.wait(0.5)

        # ── Bracket groupings ─────────────────────────────────────────────────
        # "Feature extraction" bracket over conv+pool stages (indices 1-4)
        bk_feat = Brace(
            VGroup(cuboids[1], cuboids[4]),
            direction=UP, color=C_BLUE, buff=0.55,
        )
        bk_feat_lbl = Text(
            "Feature Extraction", font_size=11, color=C_BLUE,
        ).next_to(bk_feat, UP, buff=0.08)

        # "Classifier" bracket over FC+softmax stages (indices 5-7)
        bk_cls = Brace(
            VGroup(cuboids[5], cuboids[7]),
            direction=UP, color=C_AMBER, buff=0.55,
        )
        bk_cls_lbl = Text(
            "Classifier", font_size=11, color=C_AMBER,
        ).next_to(bk_cls, UP, buff=0.08)

        self.play(
            GrowFromCenter(bk_feat), FadeIn(bk_feat_lbl),
            GrowFromCenter(bk_cls),  FadeIn(bk_cls_lbl),
            run_time=0.55,
        )
        self.wait(1.5)

        # Save everything for fade-out
        all_p1 = VGroup(
            sub, bk_feat, bk_feat_lbl, bk_cls, bk_cls_lbl,
            *cuboids, *lbl_mobs, *ann_mobs, *arrows,
        )
        self._title  = title
        self._p1_all = all_p1

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Hierarchical features
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = (
            Text(
                "Hierarchical features — deeper layers detect increasingly "
                "complex patterns",
                font_size=13, color=C_DIM,
            )
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        # ── Three-column layout ───────────────────────────────────────────────
        cols = [
            {
                "cx": -4.2,
                "header": "Early layers",
                "color":  C_BLUE,
                "features": [
                    "Horizontal edges",
                    "Vertical edges",
                    "Colour blobs",
                    "Diagonal gradients",
                ],
                "caption": "local, low-level",
            },
            {
                "cx":  0.0,
                "header": "Middle layers",
                "color":  C_PURP,
                "features": [
                    "Textures & grids",
                    "Curves & corners",
                    "Repeating patterns",
                    "Simple shapes",
                ],
                "caption": "semi-local, mid-level",
            },
            {
                "cx":  4.2,
                "header": "Late layers",
                "color":  C_AMBER,
                "features": [
                    "Object parts",
                    "Semantic regions",
                    "Viewpoint invariance",
                    "Class-specific cues",
                ],
                "caption": "global, high-level",
            },
        ]

        y_header   =  1.55
        y_feat_top =  0.90
        feat_step  =  0.52
        y_caption  = -1.55
        col_w      =  2.60

        all_col_mobs = VGroup()

        for col in cols:
            cx    = col["cx"]
            color = col["color"]

            # Header box
            hdr_box = RoundedRectangle(
                width=col_w, height=0.42, corner_radius=0.08,
                fill_color=color, fill_opacity=0.15,
                stroke_color=color, stroke_width=1.4,
            ).move_to([cx, y_header, 0])
            hdr_lbl = Text(
                col["header"], font_size=13, color=color, weight=BOLD,
            ).move_to(hdr_box)

            self.play(FadeIn(hdr_box), FadeIn(hdr_lbl), run_time=0.30)

            # Feature rows
            feat_mobs = VGroup()
            for fi, feat in enumerate(col["features"]):
                y = y_feat_top - fi * feat_step
                dot = Dot(radius=0.05, color=color).move_to([cx - 0.95, y, 0])
                txt = Text(feat, font_size=11, color=C_WHITE).move_to([cx + 0.10, y, 0])
                feat_mobs.add(VGroup(dot, txt))

            self.play(
                LaggedStart(*[FadeIn(f) for f in feat_mobs], lag_ratio=0.20),
                run_time=0.55,
            )

            # Caption
            cap = Text(col["caption"], font_size=10, color=C_DIM).move_to([cx, y_caption, 0])
            self.play(FadeIn(cap), run_time=0.22)

            all_col_mobs.add(hdr_box, hdr_lbl, feat_mobs, cap)

        # ── Arrows between columns ────────────────────────────────────────────
        arr_ab = Arrow(
            [-4.2 + col_w / 2 + 0.10,  y_header, 0],
            [ 0.0 - col_w / 2 - 0.10,  y_header, 0],
            buff=0.0, color=C_DIM, stroke_width=1.2,
            max_tip_length_to_length_ratio=0.25,
        )
        arr_bc = Arrow(
            [ 0.0 + col_w / 2 + 0.10,  y_header, 0],
            [ 4.2 - col_w / 2 - 0.10,  y_header, 0],
            buff=0.0, color=C_DIM, stroke_width=1.2,
            max_tip_length_to_length_ratio=0.25,
        )
        deeper_a = Text("deeper →", font_size=10, color=C_DIM).next_to(arr_ab, UP, buff=0.06)
        deeper_b = Text("deeper →", font_size=10, color=C_DIM).next_to(arr_bc, UP, buff=0.06)

        self.play(
            GrowArrow(arr_ab), GrowArrow(arr_bc),
            FadeIn(deeper_a), FadeIn(deeper_b),
            run_time=0.40,
        )

        caption = (
            Text(
                "same filters work for many tasks  ·  "
                "only the final classifier is task-specific",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.0)

        self._p2_all = VGroup(sub, all_col_mobs, arr_ab, arr_bc,
                              deeper_a, deeper_b, caption)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Transfer learning: freeze → fine-tune
    # ══════════════════════════════════════════════════════════════════════════

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.40)

        sub = (
            Text(
                "Transfer learning — reuse pre-trained features, "
                "train only the new head",
                font_size=13, color=C_DIM,
            )
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        # ── Layer stack ───────────────────────────────────────────────────────
        # We draw a vertical stack of layer blocks, bottom = early, top = head.
        # Frozen layers = blue;  new head layers = amber.

        BLOCK_W = 2.80
        BLOCK_H = 0.46
        GAP     = 0.10
        CX      = 0.0
        Y_BOT   = -2.55    # bottom of the stack

        frozen_layers = [
            "Conv Block 1  (edges)",
            "Conv Block 2  (textures)",
            "Conv Block 3  (shapes)",
            "Conv Block 4  (parts)",
        ]
        head_layers = [
            "FC 512  (new)",
            "Softmax  (new task)",
        ]

        n_frozen = len(frozen_layers)
        n_head   = len(head_layers)
        n_total  = n_frozen + n_head

        def _layer_block(label, color, locked, idx):
            y = Y_BOT + idx * (BLOCK_H + GAP) + BLOCK_H / 2
            rect = RoundedRectangle(
                width=BLOCK_W, height=BLOCK_H, corner_radius=0.06,
                fill_color=color, fill_opacity=0.15,
                stroke_color=color, stroke_width=1.6,
            ).move_to([CX, y, 0])
            lbl = Text(label, font_size=11, color=color).move_to([CX - 0.20, y, 0])
            mobs = VGroup(rect, lbl)
            if locked:
                lock = Text("🔒", font_size=13).move_to([CX + BLOCK_W / 2 - 0.30, y, 0])
                mobs.add(lock)
            return mobs, y

        frozen_blocks = []
        head_blocks   = []

        # ── Step 1: reveal frozen backbone ────────────────────────────────────
        step1_lbl = (
            Text("Step 1 — freeze backbone, train head only",
                 font_size=12, color=C_BLUE, weight=BOLD)
            .to_edge(LEFT, buff=0.45)
            .shift(UP * 0.60)
        )
        self.play(FadeIn(step1_lbl), run_time=0.28)

        for i, lbl in enumerate(frozen_layers):
            blk, y = _layer_block(lbl, C_BLUE, locked=True, idx=i)
            frozen_blocks.append((blk, y))
            self.play(FadeIn(blk), run_time=0.22)

        # Head layers appear with amber
        for j, lbl in enumerate(head_layers):
            blk, y = _layer_block(lbl, C_AMBER, locked=False, idx=n_frozen + j)
            head_blocks.append((blk, y))
            self.play(FadeIn(blk), run_time=0.22)

        # Gradient-flow annotation: only head trains
        grad_note = (
            Text("gradients flow here only", font_size=10, color=C_AMBER)
            .next_to(head_blocks[-1][0], RIGHT, buff=0.28)
        )
        no_grad_note = (
            Text("frozen — no gradient", font_size=10, color=C_DIM)
            .next_to(frozen_blocks[0][0], RIGHT, buff=0.28)
        )
        self.play(FadeIn(grad_note), FadeIn(no_grad_note), run_time=0.28)
        self.wait(1.2)

        # ── Step 2: unfreeze top backbone layers for fine-tuning ──────────────
        step2_lbl = (
            Text("Step 2 — unfreeze top layers, fine-tune with low LR",
                 font_size=12, color=C_GREEN, weight=BOLD)
            .next_to(step1_lbl, DOWN, buff=0.28)
            .align_to(step1_lbl, LEFT)
        )
        self.play(FadeIn(step2_lbl), run_time=0.28)

        # Unfreeze top 2 frozen layers (indices 2 and 3)
        unfreeze_idxs = [2, 3]
        for ui in unfreeze_idxs:
            old_blk = frozen_blocks[ui][0]
            y       = frozen_blocks[ui][1]
            new_blk, _ = _layer_block(
                frozen_layers[ui], C_GREEN, locked=False, idx=ui
            )
            self.play(
                FadeOut(old_blk),
                FadeIn(new_blk),
                run_time=0.35,
            )
            frozen_blocks[ui] = (new_blk, y)

        # Update gradient note
        new_grad_note = (
            Text("fine-tune with lr = 1e-5", font_size=10, color=C_GREEN)
            .next_to(frozen_blocks[unfreeze_idxs[-1]][0], RIGHT, buff=0.28)
        )
        self.play(
            FadeOut(grad_note),
            FadeIn(new_grad_note),
            run_time=0.30,
        )
        self.wait(0.6)

        # ── Legend ────────────────────────────────────────────────────────────
        leg_items = [
            (C_BLUE,  "frozen (pre-trained)"),
            (C_GREEN, "unfrozen (fine-tune)"),
            (C_AMBER, "new head (trained from scratch)"),
        ]
        legend = VGroup()
        for col, txt in leg_items:
            dot  = Dot(radius=0.07, color=col)
            ltxt = Text(txt, font_size=10, color=col)
            row  = VGroup(dot, ltxt).arrange(RIGHT, buff=0.10)
            legend.add(row)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        legend.to_edge(RIGHT, buff=0.45).shift(DOWN * 0.50)

        self.play(FadeIn(legend), run_time=0.30)

        caption = (
            Text(
                "transfer learning cuts training time and data requirements  ·  "
                "fine-tuning adapts high-level features to the new task",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.5)
