"""
W10 — RNN vs LSTM

  Phase 1: RNN unrolled across 5 time steps.
           Hidden state h_t flows right across steps; x_t feeds in
           from below. Vanishing gradient shown as magnitude bars
           shrinking during backprop.

  Phase 2: LSTM cell internals.
           Forget / input / cell-update / output gates labelled.
           Cell state C_t flows as a straight "highway" across the top.
           Contrast with RNN: short memory vs long-term cell state.

Render:
  ../../env/bin/manim -pql rnn_lstm.py RNNvsLSTM
  ../../env/bin/manim -pqh rnn_lstm.py RNNvsLSTM
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _box(label, pos, color, w=0.90, h=0.58, font_size=13):
    rect = RoundedRectangle(
        width=w, height=h, corner_radius=0.08,
        fill_color=color, fill_opacity=0.18,
        stroke_color=color, stroke_width=1.8,
    ).move_to(pos)
    txt = Text(label, font_size=font_size, color=color).move_to(pos)
    return VGroup(rect, txt)


def _arrow(start, end, color=C_DIM, stroke_w=1.6):
    return Arrow(
        start, end, buff=0.08, color=color,
        stroke_width=stroke_w,
        max_tip_length_to_length_ratio=0.22,
    )


class RNNvsLSTM(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Unrolled RNN + vanishing gradient
    # ══════════════════════════════════════════════════════════════════════════

    def _phase1(self):
        title = (
            Text("RNN  vs  LSTM", font_size=22, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.28)
        )
        sub = (
            Text(
                "RNN — same cell repeated at every time step; "
                "hidden state carries memory forward",
                font_size=13, color=C_DIM,
            )
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        T = 5
        xs_tok = ["The", "cat", "sat", "on", "mat"]
        CX0    = -4.8
        STEP   = 2.20
        Y_CELL =  0.20
        Y_IN   = -1.10
        Y_H    =  1.20

        cells   = []
        x_nodes = []
        h_nodes = []

        # ── Build cells & input nodes ─────────────────────────────────────────
        for t in range(T):
            cx = CX0 + t * STEP
            cell   = _box(f"RNN\nt={t}", [cx, Y_CELL, 0], C_BLUE,
                          w=0.88, h=0.62, font_size=11)
            x_node = _box(xs_tok[t], [cx, Y_IN, 0], C_AMBER,
                          w=0.72, h=0.38, font_size=11)
            cells.append(cell)
            x_nodes.append(x_node)

        # h_0 (initial hidden state, left of first cell)
        h0 = _box("h₀=0", [CX0 - STEP * 0.75, Y_CELL, 0], C_DIM,
                  w=0.72, h=0.38, font_size=10)

        # ── Animate cells appearing ───────────────────────────────────────────
        self.play(FadeIn(h0), run_time=0.25)
        for t in range(T):
            self.play(
                FadeIn(cells[t]), FadeIn(x_nodes[t]),
                run_time=0.28,
            )

        # ── Arrows: x_t → cell, h_{t-1} → cell, cell → h_t ─────────────────
        arr_x  = []   # x_t → cell
        arr_hh = []   # h_{t-1} → cell
        h_lbls = []   # h_t label above cell

        for t in range(T):
            cx = CX0 + t * STEP
            # x → cell
            ax = _arrow([cx, Y_IN + 0.20, 0], [cx, Y_CELL - 0.30, 0],
                        color=C_AMBER)
            arr_x.append(ax)

            # h_{t-1} → cell
            if t == 0:
                src_x = CX0 - STEP * 0.75 + 0.36
            else:
                src_x = CX0 + (t - 1) * STEP + 0.44
            ah = _arrow([src_x, Y_CELL, 0],
                        [cx - 0.44, Y_CELL, 0],
                        color=C_TEAL)
            arr_hh.append(ah)

            # h_t label above cell
            hl = Text(f"h{t+1}", font_size=11, color=C_TEAL).move_to(
                [cx + 0.44 + 0.30, Y_H, 0]
            )
            h_lbls.append(hl)

        # h arrows out of each cell (cell → right)
        arr_hout = []
        for t in range(T):
            cx = CX0 + t * STEP
            ao = _arrow([cx + 0.44, Y_CELL, 0],
                        [cx + STEP - 0.44 if t < T - 1 else cx + 0.9, Y_CELL, 0],
                        color=C_TEAL)
            arr_hout.append(ao)

        self.play(
            LaggedStart(
                *[AnimationGroup(GrowArrow(arr_x[t]),
                                 GrowArrow(arr_hh[t]),
                                 GrowArrow(arr_hout[t]),
                                 FadeIn(h_lbls[t]))
                  for t in range(T)],
                lag_ratio=0.30,
            ),
            run_time=1.8,
        )
        self.wait(0.4)

        # ── Vanishing gradient ────────────────────────────────────────────────
        vg_title = (
            Text("Backprop Through Time → vanishing gradient",
                 font_size=12, color=C_RED)
            .to_edge(DOWN, buff=1.00)
        )
        self.play(FadeIn(vg_title), run_time=0.28)

        # Gradient magnitude bars below x_nodes (right to left, shrinking)
        grad_mags  = [0.80, 0.52, 0.28, 0.12, 0.04]   # right → left
        bar_w      = 0.28
        BASELINE_Y = Y_IN - 1.10   # fixed bottom; tallest bar top stays below x_node bottom
        grad_bars  = []
        for t in range(T):
            cx   = CX0 + t * STEP
            mag  = grad_mags[T - 1 - t]     # t=0 gets smallest magnitude
            col  = interpolate_color(ManimColor(C_DEAD), ManimColor(C_RED), mag)
            h_b  = max(mag * 0.80, 0.03)
            bar  = Rectangle(
                width=bar_w, height=h_b,
                fill_color=col, fill_opacity=0.85,
                stroke_color=col, stroke_width=0.8,
            ).move_to([cx, BASELINE_Y + h_b / 2, 0])
            lbl  = Text(f"{mag:.2f}", font_size=8, color=col).next_to(bar, DOWN, buff=0.05)
            grad_bars.append(VGroup(bar, lbl))

        self.play(
            LaggedStart(*[GrowFromEdge(vg[0], DOWN) for vg in grad_bars],
                        lag_ratio=0.15),
            LaggedStart(*[FadeIn(vg[1]) for vg in grad_bars], lag_ratio=0.15),
            run_time=1.0,
        )

        vanish_note = (
            Text("early steps receive near-zero gradient → cannot learn long-range dependencies",
                 font_size=11, color=C_DIM)
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(vanish_note), run_time=0.30)
        self.wait(1.8)

        # Collect for fade-out
        self._title  = title
        self._p1_all = VGroup(
            sub, h0,
            *cells, *x_nodes, *arr_x, *arr_hh, *arr_hout, *h_lbls,
            *grad_bars, vg_title, vanish_note,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — LSTM cell internals
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = (
            Text(
                "LSTM — gating mechanism separates long-term cell state "
                "from short-term hidden state",
                font_size=13, color=C_DIM,
            )
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        # ── Layout constants ──────────────────────────────────────────────────
        CX      = 0.0
        CY      = -0.20
        GATE_W  = 1.00
        GATE_H  = 0.54
        GAP     = 1.55

        # Gate positions (left to right inside the cell)
        g_xs = [-3.0 * GAP / 2, -GAP / 2, GAP / 2, 3.0 * GAP / 2]
        # Actually spread 4 gates evenly around CX
        g_xs = [CX + (i - 1.5) * GAP for i in range(4)]
        gate_defs = [
            ("forget\ngate  σ",  C_RED,   g_xs[0]),
            ("input\ngate  σ",   C_BLUE,  g_xs[1]),
            ("cell\nupdate  tanh", C_PURP, g_xs[2]),
            ("output\ngate  σ",  C_GREEN, g_xs[3]),
        ]

        # ── Cell state highway (top) ──────────────────────────────────────────
        Y_HIGHWAY = CY + 1.30
        Y_GATE    = CY
        Y_INPUT   = CY - 1.25

        highway_line = DashedLine(
            [-5.5, Y_HIGHWAY, 0], [5.5, Y_HIGHWAY, 0],
            dash_length=0.18, stroke_color=C_GOLD, stroke_width=2.2,
        )
        hw_lbl = Text("Cell state  C_t  (long-term memory)",
                      font_size=11, color=C_GOLD).move_to([0.0, Y_HIGHWAY + 0.32, 0])
        c_in  = Text("C_{t-1}", font_size=11, color=C_GOLD).move_to([-4.5, Y_HIGHWAY - 0.32, 0])
        c_out = Text("C_t",     font_size=11, color=C_GOLD).move_to([ 4.5, Y_HIGHWAY - 0.32, 0])

        self.play(
            Create(highway_line), FadeIn(hw_lbl),
            FadeIn(c_in), FadeIn(c_out),
            run_time=0.55,
        )

        # ── Gates ─────────────────────────────────────────────────────────────
        gate_mobs = []
        for label, color, gx in gate_defs:
            g = _box(label, [gx, Y_GATE, 0], color,
                     w=GATE_W, h=GATE_H, font_size=10)
            gate_mobs.append(g)

        self.play(
            LaggedStart(*[FadeIn(g) for g in gate_mobs], lag_ratio=0.20),
            run_time=0.70,
        )

        # ── Input arrows (x_t + h_{t-1} → each gate) ─────────────────────────
        x_in_lbl  = Text("x_t",     font_size=12, color=C_AMBER).move_to([-1.20, Y_INPUT, 0])
        h_in_lbl  = Text("h_{t-1}", font_size=12, color=C_TEAL ).move_to([ 1.20, Y_INPUT, 0])
        h_out_lbl = Text("h_t",     font_size=12, color=C_TEAL ).move_to([ 5.0,  Y_GATE,  0])

        self.play(FadeIn(x_in_lbl), FadeIn(h_in_lbl), FadeIn(h_out_lbl), run_time=0.28)

        in_arrows = VGroup()
        for _, _, gx in gate_defs:
            in_arrows.add(
                _arrow([gx - 0.20, Y_INPUT + 0.25, 0],
                       [gx - 0.20, Y_GATE  - 0.28, 0], color=C_AMBER)
            )
            in_arrows.add(
                _arrow([gx + 0.20, Y_INPUT + 0.25, 0],
                       [gx + 0.20, Y_GATE  - 0.28, 0], color=C_TEAL)
            )

        self.play(
            LaggedStart(*[GrowArrow(a) for a in in_arrows], lag_ratio=0.08),
            run_time=0.70,
        )

        # ── Gate → highway arrows ─────────────────────────────────────────────
        hw_arrows = VGroup()
        for _, color, gx in gate_defs:
            hw_arrows.add(
                _arrow([gx, Y_GATE + GATE_H / 2 + 0.05, 0],
                       [gx, Y_HIGHWAY - 0.05, 0], color=color)
            )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in hw_arrows], lag_ratio=0.20),
            run_time=0.60,
        )

        # ── Annotations ───────────────────────────────────────────────────────
        # Annotations placed well below where the input arrows start (Y_INPUT+0.25)
        ann_y = Y_INPUT - 0.55
        ann_forget = Text("what to forget\nfrom C_{t-1}", font_size=9, color=C_RED  ).move_to([g_xs[0], ann_y, 0])
        ann_input  = Text("what new info\nto write",       font_size=9, color=C_BLUE ).move_to([g_xs[1], ann_y, 0])
        ann_cell   = Text("candidate\nvalues",             font_size=9, color=C_PURP ).move_to([g_xs[2], ann_y, 0])
        ann_out    = Text("what to\nexpose as h_t",        font_size=9, color=C_GREEN).move_to([g_xs[3], ann_y, 0])

        self.play(
            LaggedStart(
                FadeIn(ann_forget), FadeIn(ann_input),
                FadeIn(ann_cell),   FadeIn(ann_out),
                lag_ratio=0.20,
            ),
            run_time=0.65,
        )

        # ── Contrast caption ──────────────────────────────────────────────────
        caption = (
            Text(
                "cell state flows with minimal modification  ·  "
                "gates learn when to remember, forget, and output",
                font_size=12, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.5)
