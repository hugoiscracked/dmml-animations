"""
W09 — Dropout & Batch Normalization

  Phase 1: Dropout — a 4-node hidden layer with neurons randomly dropped
           during training (red X marks). Side-by-side: training (sparse)
           vs inference (all active, weights scaled by keep-prob).
  Phase 2: Overfitting context — two loss curve pairs showing train vs
           validation: without dropout (val diverges) and with dropout
           (both converge). Illustrates *why* dropout helps.
  Phase 3: Batch Normalization — a column of raw activations (spread,
           shifted) is normalised to mean≈0 / std≈1, then scaled and
           shifted by learned gamma/beta. Before/after bar distributions.

Render:
  ../../env/bin/manim -pql dropout_batchnorm.py DropoutBatchNorm
  ../../env/bin/manim -pqh dropout_batchnorm.py DropoutBatchNorm
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
C_DEAD  = "#3d444d"

NODE_R  = 0.34

# ── Shared node / edge helpers ─────────────────────────────────────────────────

def _node(pos, color, fill_op=0.18, r=NODE_R):
    return (
        Circle(radius=r, color=color, fill_color=color,
               fill_opacity=fill_op, stroke_width=2.0)
        .move_to(pos)
    )


def _edge(p1, p2, color=C_DIM, width=1.0):
    return Line(np.array(p1), np.array(p2),
                stroke_color=color, stroke_width=width)


def _cross(pos, color=C_RED, scale=0.28):
    """Return a VGroup X mark centred on pos."""
    a = Line([-scale, -scale, 0], [scale,  scale, 0],
             stroke_color=color, stroke_width=3.0)
    b = Line([-scale,  scale, 0], [scale, -scale, 0],
             stroke_color=color, stroke_width=3.0)
    return VGroup(a, b).move_to(pos)


class DropoutBatchNorm(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ── Phase 1: Dropout ──────────────────────────────────────────────────────

    def _phase1(self):
        title = (
            Text("Dropout  &  Batch Normalization",
                 font_size=22, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.28)
        )
        sub = (
            Text("Dropout — randomly deactivate neurons during training to prevent co-adaptation",
                 font_size=13, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        # ── Layout: two panels (training | inference) ─────────────────────────
        # Each panel: input layer (3 nodes) → hidden layer (4 nodes) → output (1 node)

        # Panel centres
        L_CX = -3.3   # left panel (training) centre x
        R_CX =  3.3   # right panel (inference) centre x

        def _panel(cx, dropped=None):
            """
            Build one mini-network panel.
            dropped: list of hidden-node indices to mark as dropped (0-based).
            Returns dict of VGroups.
            """
            # x positions
            x_in  = cx - 1.8
            x_hid = cx
            x_out = cx + 1.8

            # y positions
            y_in  = [0.8, 0.0, -0.8]
            y_hid = [1.2, 0.4, -0.4, -1.2]
            y_out = [0.0]

            in_nodes  = VGroup(*[_node([x_in,  y, 0], C_BLUE)  for y in y_in])
            hid_nodes = VGroup()
            for i, y in enumerate(y_hid):
                col = C_DEAD if (dropped and i in dropped) else C_AMBER
                fo  = 0.55   if (dropped and i in dropped) else 0.20
                hid_nodes.add(_node([x_hid, y, 0], col, fill_op=fo))

            out_nodes = VGroup(*[_node([x_out, y, 0], C_GREEN) for y in y_out])

            # Edges: input → hidden
            edges_ih = VGroup()
            for i, yi in enumerate(y_in):
                for j, yj in enumerate(y_hid):
                    if dropped and j in dropped:
                        continue   # no edges to dropped nodes
                    edges_ih.add(_edge([x_in + NODE_R, yi, 0],
                                       [x_hid - NODE_R, yj, 0],
                                       color=C_DIM, width=0.7))

            # Edges: hidden → output
            edges_ho = VGroup()
            for j, yj in enumerate(y_hid):
                if dropped and j in dropped:
                    continue
                edges_ho.add(_edge([x_hid + NODE_R, yj, 0],
                                   [x_out - NODE_R, y_out[0], 0],
                                   color=C_DIM, width=0.7))

            return dict(
                in_nodes=in_nodes, hid_nodes=hid_nodes, out_nodes=out_nodes,
                edges_ih=edges_ih, edges_ho=edges_ho,
                positions=dict(hid=[[x_hid, y, 0] for y in y_hid]),
            )

        DROPPED = [1, 3]   # nodes 1 and 3 are dropped in training panel

        train_p = _panel(L_CX, dropped=DROPPED)
        infer_p = _panel(R_CX, dropped=None)

        # ── Divider ───────────────────────────────────────────────────────────
        divider = DashedLine(
            [0, 2.8, 0], [0, -2.8, 0],
            dash_length=0.12, color=C_DIM, stroke_width=0.8,
        )

        # ── Panel labels ──────────────────────────────────────────────────────
        lbl_train = Text("Training  (p = 0.5)", font_size=13, color=C_AMBER,
                         weight=BOLD).move_to([L_CX, 2.3, 0])
        lbl_infer = Text("Inference  (all active)", font_size=13, color=C_GREEN,
                         weight=BOLD).move_to([R_CX, 2.3, 0])

        # ── Animate training panel ────────────────────────────────────────────
        self.play(FadeIn(divider), FadeIn(lbl_train), FadeIn(lbl_infer),
                  run_time=0.40)

        self.play(
            LaggedStart(*[GrowFromCenter(n) for n in train_p["in_nodes"]],
                        lag_ratio=0.15),
            run_time=0.45,
        )
        self.play(
            LaggedStart(*[Create(e) for e in train_p["edges_ih"]], lag_ratio=0.05),
            run_time=0.50,
        )
        self.play(
            LaggedStart(*[GrowFromCenter(n) for n in train_p["hid_nodes"]],
                        lag_ratio=0.12),
            run_time=0.45,
        )
        self.play(
            LaggedStart(*[Create(e) for e in train_p["edges_ho"]], lag_ratio=0.07),
            run_time=0.35,
        )
        self.play(GrowFromCenter(train_p["out_nodes"][0]), run_time=0.28)

        # X marks on dropped nodes
        crosses = VGroup(*[
            _cross(train_p["positions"]["hid"][i])
            for i in DROPPED
        ])
        self.play(
            LaggedStart(*[GrowFromCenter(x) for x in crosses], lag_ratio=0.3),
            run_time=0.45,
        )

        drop_note = (
            Text("dropped  (output = 0)", font_size=10, color=C_RED)
            .next_to(train_p["hid_nodes"], RIGHT, buff=0.15)
        )
        self.play(FadeIn(drop_note), run_time=0.22)
        self.wait(0.4)

        # ── Animate inference panel ───────────────────────────────────────────
        self.play(
            LaggedStart(*[GrowFromCenter(n) for n in infer_p["in_nodes"]],
                        lag_ratio=0.12),
            run_time=0.40,
        )
        self.play(
            LaggedStart(*[Create(e) for e in infer_p["edges_ih"]], lag_ratio=0.04),
            run_time=0.45,
        )
        self.play(
            LaggedStart(*[GrowFromCenter(n) for n in infer_p["hid_nodes"]],
                        lag_ratio=0.12),
            run_time=0.40,
        )
        self.play(
            LaggedStart(*[Create(e) for e in infer_p["edges_ho"]], lag_ratio=0.06),
            run_time=0.30,
        )
        self.play(GrowFromCenter(infer_p["out_nodes"][0]), run_time=0.25)

        scale_note = (
            Text("weights scaled by (1 - p)", font_size=10, color=C_BLUE)
            .next_to(infer_p["hid_nodes"], LEFT, buff=0.15)
        )
        self.play(FadeIn(scale_note), run_time=0.22)

        caption = (
            Text(
                "each training step uses a different random sub-network  "
                "→  ensemble effect at test time",
                font_size=11, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(1.8)

        self._title  = title
        self._p1_all = VGroup(
            sub, divider, lbl_train, lbl_infer,
            train_p["in_nodes"], train_p["hid_nodes"], train_p["out_nodes"],
            train_p["edges_ih"], train_p["edges_ho"], crosses, drop_note,
            infer_p["in_nodes"], infer_p["hid_nodes"], infer_p["out_nodes"],
            infer_p["edges_ih"], infer_p["edges_ho"], scale_note,
            caption,
        )

    # ── Phase 2: loss curves ──────────────────────────────────────────────────

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = (
            Text("effect on training: dropout reduces the train/validation gap",
                 font_size=13, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        epochs = np.arange(1, 31)

        def _loss(base, decay, noise_amp, seed):
            rng = np.random.default_rng(seed)
            return base * np.exp(-decay * epochs) + noise_amp * rng.random(len(epochs))

        # Without dropout
        train_nd = _loss(1.8, 0.14, 0.04, 1)   # drives to ~0.05
        val_nd   = _loss(1.6, 0.06, 0.05, 2) + 0.25   # plateaus high → overfit

        # With dropout
        train_do = _loss(1.8, 0.10, 0.05, 3) + 0.05   # slightly slower train
        val_do   = _loss(1.6, 0.09, 0.04, 4) + 0.05   # tracks training

        # Clamp
        train_nd = np.clip(train_nd, 0.04, 2.0)
        val_nd   = np.clip(val_nd,   0.30, 2.0)
        train_do = np.clip(train_do, 0.18, 2.0)
        val_do   = np.clip(val_do,   0.20, 2.0)

        panel_specs = [
            ("Without Dropout", train_nd, val_nd,   C_RED,   -3.4),
            ("With Dropout",    train_do, val_do,   C_GREEN,  3.4),
        ]

        ax_w, ax_h = 4.4, 2.4
        y_top = 1.0

        all_mobs = VGroup()

        for title_txt, t_loss, v_loss, col, cx in panel_specs:
            ax = Axes(
                x_range=[0, 31, 10],
                y_range=[0, 2.2, 0.5],
                x_length=ax_w,
                y_length=ax_h,
                axis_config={"color": C_DIM, "stroke_width": 1.2,
                             "include_ticks": False},
            ).move_to([cx, y_top - ax_h / 2, 0])

            # Axis labels (manual Text, no LaTeX)
            x_lbl = Text("Epochs", font_size=10, color=C_DIM).next_to(ax, DOWN, buff=0.08)
            y_lbl = (
                Text("Loss", font_size=10, color=C_DIM)
                .rotate(PI / 2)
                .next_to(ax, LEFT, buff=0.08)
            )
            panel_lbl = (
                Text(title_txt, font_size=13, color=col, weight=BOLD)
                .next_to(ax, UP, buff=0.12)
            )

            self.play(Create(ax), FadeIn(x_lbl), FadeIn(y_lbl),
                      FadeIn(panel_lbl), run_time=0.45)

            # Build point lists
            def _pts(arr):
                return [ax.c2p(e, arr[i]) for i, e in enumerate(epochs)]

            train_pts = _pts(t_loss)
            val_pts   = _pts(v_loss)

            train_path = VMobject(stroke_color=C_BLUE,  stroke_width=2.0)
            val_path   = VMobject(stroke_color=C_AMBER, stroke_width=2.0)
            train_path.set_points_as_corners(train_pts)
            val_path.set_points_as_corners(val_pts)

            self.play(Create(train_path), run_time=0.7)
            self.play(Create(val_path),   run_time=0.7)

            # Legend
            leg_train = VGroup(
                Line([0, 0, 0], [0.3, 0, 0], stroke_color=C_BLUE,  stroke_width=2.0),
                Text("train", font_size=10, color=C_BLUE).shift(RIGHT * 0.5),
            ).arrange(RIGHT, buff=0.08).next_to(ax, DOWN, buff=0.28).shift(LEFT * 0.5)
            leg_val = VGroup(
                Line([0, 0, 0], [0.3, 0, 0], stroke_color=C_AMBER, stroke_width=2.0),
                Text("val", font_size=10, color=C_AMBER).shift(RIGHT * 0.5),
            ).arrange(RIGHT, buff=0.08).next_to(leg_train, RIGHT, buff=0.35)

            self.play(FadeIn(leg_train), FadeIn(leg_val), run_time=0.22)
            all_mobs.add(ax, x_lbl, y_lbl, panel_lbl,
                         train_path, val_path, leg_train, leg_val)

        # Gap annotation on left panel
        gap_note = (
            Text("large gap = overfitting", font_size=11, color=C_RED)
            .move_to([-3.4, -1.85, 0])
        )
        close_note = (
            Text("gap closed by dropout", font_size=11, color=C_GREEN)
            .move_to([3.4, -1.85, 0])
        )
        self.play(FadeIn(gap_note), FadeIn(close_note), run_time=0.30)
        self.wait(2.0)

        self._p2_all = VGroup(all_mobs, sub, gap_note, close_note)

    # ── Phase 3: Batch Normalization ──────────────────────────────────────────

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.40)

        sub = (
            Text("Batch Normalization — normalise activations within each mini-batch",
                 font_size=13, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        # Raw activation values (hand-picked for visual drama)
        raw_vals = np.array([8.5, 2.1, 11.3, -1.4, 6.0, 14.2, 0.3, 4.8])
        mu  = raw_vals.mean()
        sig = raw_vals.std() + 1e-5
        norm_vals = (raw_vals - mu) / sig

        # Learned params (make it interesting: gamma=1.5, beta=0.5)
        gamma, beta = 1.5, 0.5
        out_vals = gamma * norm_vals + beta

        # ── Bar chart helper ──────────────────────────────────────────────────
        N        = len(raw_vals)
        bar_w    = 0.30
        bar_gap  = 0.12
        step     = bar_w + bar_gap

        def _bar_chart(values, origin, color, y_scale=0.12, label=None):
            """Return VGroup of bars + optional label."""
            bars = VGroup()
            for i, v in enumerate(values):
                h = abs(v) * y_scale
                h = max(h, 0.04)
                bar = Rectangle(
                    width=bar_w, height=h,
                    fill_color=color, fill_opacity=0.75,
                    stroke_color=color, stroke_width=0.8,
                )
                bx = origin[0] + i * step
                by = origin[1] + (h / 2 if v >= 0 else -h / 2)
                bar.move_to([bx, by, 0])
                bars.add(bar)
            grp = VGroup(bars)
            if label:
                lbl = Text(label, font_size=11, color=color).next_to(bars, UP, buff=0.12)
                grp.add(lbl)
            return grp

        # x-origin so each chart is centred
        def _cx_origin(cx, n):
            return cx - (n - 1) * step / 2

        raw_origin  = [_cx_origin(-4.6, N), -0.5, 0.0]
        norm_origin = [_cx_origin( 0.0, N), -0.5, 0.0]
        out_origin  = [_cx_origin( 4.6, N), -0.5, 0.0]

        raw_chart  = _bar_chart(raw_vals,  raw_origin,  C_RED,   y_scale=0.10,
                                label="Raw activations")
        norm_chart = _bar_chart(norm_vals, norm_origin, C_BLUE,  y_scale=0.45,
                                label="After normalise")
        out_chart  = _bar_chart(out_vals,  out_origin,  C_GREEN, y_scale=0.30,
                                label="After  gamma*x + beta")

        # Zero lines
        def _zero_line(origin, n, color):
            x0 = origin[0] - step / 2
            x1 = origin[0] + (n - 1) * step + step / 2
            return DashedLine([x0, origin[1], 0], [x1, origin[1], 0],
                              dash_length=0.10, stroke_color=color,
                              stroke_width=0.8)

        z0_raw  = _zero_line(raw_origin,  N, C_DIM)
        z0_norm = _zero_line(norm_origin, N, C_DIM)
        z0_out  = _zero_line(out_origin,  N, C_DIM)

        # ── Arrows between stages ─────────────────────────────────────────────
        arr1 = Arrow([-2.35, -0.5, 0], [-1.35, -0.5, 0],
                     buff=0.05, color=C_DIM, stroke_width=1.4,
                     max_tip_length_to_length_ratio=0.22)
        arr2 = Arrow([ 1.35, -0.5, 0], [ 2.35, -0.5, 0],
                     buff=0.05, color=C_DIM, stroke_width=1.4,
                     max_tip_length_to_length_ratio=0.22)

        # ── Step labels below charts ──────────────────────────────────────────
        def _step_lbl(text, cx, color):
            return Text(text, font_size=10, color=color).move_to([cx, -2.10, 0])

        s1 = _step_lbl("x  (any scale)", -4.6, C_RED)
        s2 = _step_lbl("x_hat = (x - mu) / sigma", 0.0, C_BLUE)
        s3 = _step_lbl("y = gamma * x_hat + beta", 4.6, C_GREEN)

        # ── Formula lines ─────────────────────────────────────────────────────
        f_norm = (
            Text(f"mu = {mu:.1f}   sigma = {sig:.1f}",
                 font_size=11, color=C_DIM)
            .move_to([0.0, -2.55, 0])
        )
        f_out = (
            Text(f"gamma = {gamma}   beta = {beta}   (learned)",
                 font_size=11, color=C_DIM)
            .move_to([4.6, -2.55, 0])
        )

        # ── Animate ───────────────────────────────────────────────────────────
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in raw_chart[0]],
                        lag_ratio=0.07),
            FadeIn(z0_raw),
            run_time=0.65,
        )
        self.play(FadeIn(raw_chart[1]), FadeIn(s1), run_time=0.25)

        self.play(GrowArrow(arr1), run_time=0.25)

        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in norm_chart[0]],
                        lag_ratio=0.07),
            FadeIn(z0_norm),
            run_time=0.65,
        )
        self.play(FadeIn(norm_chart[1]), FadeIn(s2), FadeIn(f_norm), run_time=0.28)
        self.wait(0.35)

        self.play(GrowArrow(arr2), run_time=0.25)

        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in out_chart[0]],
                        lag_ratio=0.07),
            FadeIn(z0_out),
            run_time=0.65,
        )
        self.play(FadeIn(out_chart[1]), FadeIn(s3), FadeIn(f_out), run_time=0.28)

        caption = (
            Text(
                "BatchNorm stabilises training  ·  "
                "allows higher learning rates  ·  "
                "mild regularisation effect",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.30)
        )
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.5)
