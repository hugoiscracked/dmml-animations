"""
W01 — Descriptive Statistics Overview

  Phase 1: Central tendency & spread.
           13 data points on a number line; mean (amber) and median
           (teal) lines appear with labels, followed by a ±1σ brace
           showing how an outlier pulls the mean away from the median.

  Phase 2: Box plot — built step by step.
           Same data set: Q1/Q3 ticks → IQR box → median line →
           whiskers → outlier dot.  Each element is labelled as it
           appears.

  Phase 3: Skewness.
           Three histograms (symmetric / right-skewed / left-skewed)
           show how asymmetry shifts the mean relative to the median.

Render:
  ../../env/bin/manim -pql stats_overview.py StatsOverview
  ../../env/bin/manim -pqh stats_overview.py StatsOverview
"""

from manim import *
import numpy as np

# ── Palette ────────────────────────────────────────────────────────────────────
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

# ── Shared dataset ─────────────────────────────────────────────────────────────
DATA = np.array([2., 5., 7., 8., 9., 10., 11., 12., 14., 17., 18., 22., 35.])
MEAN = float(np.mean(DATA))          # ≈ 13.1
MED  = float(np.median(DATA))        # 11.0
Q1   = float(np.percentile(DATA, 25))  # 8.0
Q3   = float(np.percentile(DATA, 75))  # 17.0
IQR  = Q3 - Q1
STD  = float(np.std(DATA))           # ≈ 8.2
WLO  = float(DATA[DATA >= Q1 - 1.5 * IQR].min())   # 2.0
WHI  = float(DATA[DATA <= Q3 + 1.5 * IQR].max())   # 22.0
OUTS = DATA[DATA > Q3 + 1.5 * IQR]                 # [35]

# Axis mapping: data [0, 38] → screen x [-3.5, 3.5]
AX_L, AX_R = -3.5, 3.5

def to_x(v):
    return AX_L + (v / 38.0) * (AX_R - AX_L)


class StatsOverview(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Central tendency & spread
    # ══════════════════════════════════════════════════════════════════════════

    def _phase1(self):
        title = (
            Text("Descriptive Statistics", font_size=24,
                 color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.25)
        )
        sub = (
            Text("central tendency and spread — mean, median, standard deviation",
                 font_size=13, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        AXIS_Y = 0.30

        # ── Axis ──────────────────────────────────────────────────────────────
        axis = Line([AX_L, AXIS_Y, 0], [AX_R, AXIS_Y, 0],
                    stroke_color=C_DIM, stroke_width=1.2)
        ticks = VGroup()
        for v, lbl in [(0, "0"), (10, "10"), (20, "20"), (30, "30")]:
            tx = to_x(v)
            ticks.add(Line([tx, AXIS_Y - 0.10, 0], [tx, AXIS_Y + 0.10, 0],
                           stroke_color=C_DIM, stroke_width=0.8))
            ticks.add(Text(lbl, font_size=8, color=C_DIM)
                      .move_to([tx, AXIS_Y - 0.26, 0]))

        self.play(Create(axis), FadeIn(ticks), run_time=0.40)

        # ── Data dots ─────────────────────────────────────────────────────────
        dots = VGroup(*[
            Dot([to_x(v), AXIS_Y, 0], radius=0.08,
                fill_color=C_BLUE, fill_opacity=0.85)
            for v in DATA
        ])
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.10),
            run_time=0.80,
        )
        self.wait(0.3)

        # ── Mean line ─────────────────────────────────────────────────────────
        mean_x = to_x(MEAN)
        mean_line = DashedLine(
            [mean_x, AXIS_Y - 0.35, 0], [mean_x, AXIS_Y + 1.00, 0],
            dash_length=0.10, stroke_color=C_AMBER, stroke_width=2.0,
        )
        mean_lbl = Text(f"mean = {MEAN:.1f}", font_size=11, color=C_AMBER,
                        weight=BOLD).move_to([mean_x + 0.85, AXIS_Y + 1.18, 0])
        self.play(Create(mean_line), FadeIn(mean_lbl), run_time=0.40)

        # ── Median line ───────────────────────────────────────────────────────
        med_x = to_x(MED)
        med_line = DashedLine(
            [med_x, AXIS_Y - 0.35, 0], [med_x, AXIS_Y + 1.00, 0],
            dash_length=0.10, stroke_color=C_TEAL, stroke_width=2.0,
        )
        med_lbl = Text(f"median = {MED:.1f}", font_size=11, color=C_TEAL,
                       weight=BOLD).move_to([med_x - 1.00, AXIS_Y + 1.18, 0])
        self.play(Create(med_line), FadeIn(med_lbl), run_time=0.40)

        # Outlier callout
        out_x  = to_x(OUTS[0])
        out_arr = Arrow(
            [out_x, AXIS_Y + 0.55, 0], [out_x, AXIS_Y + 0.12, 0],
            buff=0.0, color=C_RED,
            stroke_width=1.4, max_tip_length_to_length_ratio=0.22,
        )
        out_lbl = Text(f"outlier ({int(OUTS[0])})", font_size=10, color=C_RED)\
            .next_to(out_arr, UP, buff=0.06)
        self.play(GrowArrow(out_arr), FadeIn(out_lbl), run_time=0.35)
        self.wait(0.3)

        # ── ±1σ brace ─────────────────────────────────────────────────────────
        std_lo_x = to_x(MEAN - STD)
        std_hi_x = to_x(MEAN + STD)
        std_ref  = Line([std_lo_x, AXIS_Y - 0.45, 0],
                        [std_hi_x, AXIS_Y - 0.45, 0])
        std_brace = Brace(std_ref, DOWN, color=C_BLUE, buff=0.04)
        std_lbl   = Text(f"mean ± σ  (σ = {STD:.1f})", font_size=10, color=C_BLUE)\
            .next_to(std_brace, DOWN, buff=0.06)
        self.play(Create(std_brace), FadeIn(std_lbl), run_time=0.45)

        caption = (
            Text("mean is pulled by the outlier  ·  median is more robust",
                 font_size=11, color=C_DIM)
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.8)

        self._title  = title
        self._p1_all = VGroup(
            sub, axis, ticks, dots,
            mean_line, mean_lbl, med_line, med_lbl,
            out_arr, out_lbl, std_ref, std_brace, std_lbl, caption,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Box plot
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = (
            Text("Box plot — five-number summary built layer by layer",
                 font_size=13, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        DOT_Y = 1.10
        BOX_Y = 0.00
        BOX_H = 0.50

        # ── Mini axis + dots ──────────────────────────────────────────────────
        axis = Line([AX_L, DOT_Y, 0], [AX_R, DOT_Y, 0],
                    stroke_color=C_DIM, stroke_width=1.0)
        dots = VGroup(*[
            Dot([to_x(v), DOT_Y, 0], radius=0.07,
                fill_color=C_BLUE if v not in OUTS else C_RED,
                fill_opacity=0.85)
            for v in DATA
        ])
        self.play(Create(axis), run_time=0.25)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.08),
            run_time=0.65,
        )
        self.wait(0.2)

        # ── Q1 / Q3 ticks ─────────────────────────────────────────────────────
        q1x, q3x = to_x(Q1), to_x(Q3)

        def vtick(x, y_mid, h, color):
            return Line([x, y_mid - h / 2, 0], [x, y_mid + h / 2, 0],
                        stroke_color=color, stroke_width=2.0)

        q1_tick = vtick(q1x, BOX_Y, BOX_H, C_PURP)
        q3_tick = vtick(q3x, BOX_Y, BOX_H, C_PURP)
        q1_lbl  = Text(f"Q1={int(Q1)}", font_size=10, color=C_PURP)\
            .move_to([q1x, BOX_Y - BOX_H / 2 - 0.28, 0])
        q3_lbl  = Text(f"Q3={int(Q3)}", font_size=10, color=C_PURP)\
            .move_to([q3x, BOX_Y - BOX_H / 2 - 0.28, 0])
        self.play(
            Create(q1_tick), Create(q3_tick),
            FadeIn(q1_lbl), FadeIn(q3_lbl),
            run_time=0.40,
        )

        # ── IQR box ───────────────────────────────────────────────────────────
        box_cx = (q1x + q3x) / 2
        box_w  = q3x - q1x
        box = Rectangle(
            width=box_w, height=BOX_H,
            fill_color=C_PURP, fill_opacity=0.18,
            stroke_color=C_PURP, stroke_width=1.6,
        ).move_to([box_cx, BOX_Y, 0])
        iqr_ref   = Line([q1x, BOX_Y - BOX_H/2 - 0.55, 0],
                         [q3x, BOX_Y - BOX_H/2 - 0.55, 0])
        iqr_brace = Brace(iqr_ref, DOWN, color=C_PURP, buff=0.04)
        iqr_lbl   = Text(f"IQR = {int(IQR)}", font_size=10, color=C_PURP)\
            .next_to(iqr_brace, DOWN, buff=0.05)
        self.play(Create(box), Create(iqr_brace), FadeIn(iqr_lbl), run_time=0.45)

        # ── Median line ───────────────────────────────────────────────────────
        med_x = to_x(MED)
        med_line = Line([med_x, BOX_Y - BOX_H/2, 0], [med_x, BOX_Y + BOX_H/2, 0],
                        stroke_color=C_TEAL, stroke_width=2.4)
        med_lbl  = Text(f"median = {int(MED)}", font_size=10, color=C_TEAL)\
            .move_to([med_x, BOX_Y + BOX_H/2 + 0.28, 0])
        self.play(Create(med_line), FadeIn(med_lbl), run_time=0.35)

        # ── Whiskers ──────────────────────────────────────────────────────────
        wlo_x, whi_x = to_x(WLO), to_x(WHI)
        w_lo  = Line([wlo_x, BOX_Y, 0], [q1x,  BOX_Y, 0],
                     stroke_color=C_DIM, stroke_width=1.4)
        w_hi  = Line([q3x,  BOX_Y, 0], [whi_x, BOX_Y, 0],
                     stroke_color=C_DIM, stroke_width=1.4)
        cap_lo = vtick(wlo_x, BOX_Y, 0.20, C_DIM)
        cap_hi = vtick(whi_x, BOX_Y, 0.20, C_DIM)
        wlo_lbl = Text(f"min={int(WLO)}", font_size=9, color=C_DIM)\
            .move_to([wlo_x, BOX_Y + BOX_H/2 + 0.28, 0])
        whi_lbl = Text(f"max={int(WHI)}", font_size=9, color=C_DIM)\
            .move_to([whi_x, BOX_Y + BOX_H/2 + 0.28, 0])
        self.play(
            Create(w_lo), Create(w_hi),
            Create(cap_lo), Create(cap_hi),
            FadeIn(wlo_lbl), FadeIn(whi_lbl),
            run_time=0.45,
        )

        # ── Outlier ───────────────────────────────────────────────────────────
        out_x   = to_x(OUTS[0])
        out_dot = Dot([out_x, BOX_Y, 0], radius=0.11,
                      fill_color=C_RED, fill_opacity=0.90)
        out_lbl = Text(f"outlier ({int(OUTS[0])})", font_size=10, color=C_RED)\
            .move_to([out_x, BOX_Y + BOX_H/2 + 0.28, 0])
        out_note = Text(f"> Q3 + 1.5×IQR = {Q3 + 1.5*IQR:.0f}", font_size=9,
                        color=C_RED).move_to([out_x, BOX_Y - BOX_H/2 - 0.28, 0])
        self.play(GrowFromCenter(out_dot), FadeIn(out_lbl), FadeIn(out_note),
                  run_time=0.40)
        self.wait(1.8)

        self._p2_all = VGroup(
            sub, axis, dots,
            q1_tick, q3_tick, q1_lbl, q3_lbl,
            box, iqr_ref, iqr_brace, iqr_lbl,
            med_line, med_lbl,
            w_lo, w_hi, cap_lo, cap_hi, wlo_lbl, whi_lbl,
            out_dot, out_lbl, out_note,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Skewness
    # ══════════════════════════════════════════════════════════════════════════

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.40)

        sub = (
            Text("Skewness — asymmetry shifts the mean away from the median",
                 font_size=13, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        BASELINE_Y = -1.40
        BAR_MAX_H  =  2.20
        BAR_W      =  0.38
        STEP       =  0.48
        N_BINS     =  7

        def bin_centers(cx):
            return np.array([cx + (i - (N_BINS - 1) / 2) * STEP
                             for i in range(N_BINS)])

        def weighted_mean(cx, heights):
            xs = bin_centers(cx)
            h  = np.array(heights)
            return float(np.sum(xs * h) / np.sum(h))

        def weighted_median(cx, heights):
            xs  = bin_centers(cx)
            h   = np.array(heights)
            cum = np.cumsum(h)
            half = cum[-1] / 2
            for i in range(N_BINS):
                if cum[i] >= half:
                    prev = cum[i - 1] if i > 0 else 0.0
                    frac = (half - prev) / (cum[i] - prev)
                    return float(xs[i] - STEP / 2 + frac * STEP)
            return float(xs[-1])

        # Histogram definitions
        h_sym  = [0.15, 0.55, 0.95, 1.00, 0.80, 0.40, 0.12]
        h_rsk  = [1.00, 0.78, 0.54, 0.32, 0.18, 0.09, 0.04]
        h_lsk  = list(reversed(h_rsk))

        configs = [
            (-4.20, h_sym,  C_BLUE,  "Symmetric",          C_BLUE ),
            ( 0.00, h_rsk,  C_AMBER, "Right-skewed  (+)",   C_AMBER),
            ( 4.20, h_lsk,  C_PURP,  "Left-skewed  (−)",    C_PURP ),
        ]

        all_groups = VGroup()

        for cx, heights, bar_col, title_str, title_col in configs:
            grp = VGroup()

            # Baseline
            n   = len(heights)
            x0  = cx - (n - 1) * STEP / 2
            grp.add(Line(
                [x0 - BAR_W / 2 - 0.06, BASELINE_Y, 0],
                [x0 + (n - 1) * STEP + BAR_W / 2 + 0.06, BASELINE_Y, 0],
                stroke_color=C_DIM, stroke_width=1.0,
            ))

            # Title
            grp.add(
                Text(title_str, font_size=12, color=title_col, weight=BOLD)
                .move_to([cx, BASELINE_Y + BAR_MAX_H + 0.40, 0])
            )

            # Bars
            bars = []
            for i, h_rel in enumerate(heights):
                h   = max(h_rel * BAR_MAX_H, 0.04)
                bx  = x0 + i * STEP
                bar = Rectangle(
                    width=BAR_W, height=h,
                    fill_color=bar_col, fill_opacity=0.72,
                    stroke_color=bar_col, stroke_width=0.6,
                ).move_to([bx, BASELINE_Y + h / 2, 0])
                bars.append(bar)

            # Mean / median vertical lines
            mean_x = weighted_mean(cx, heights)
            med_x  = weighted_median(cx, heights)

            line_top = BASELINE_Y + BAR_MAX_H + 0.10
            line_bot = BASELINE_Y

            mean_line = DashedLine(
                [mean_x, line_bot, 0], [mean_x, line_top, 0],
                dash_length=0.09, stroke_color=C_AMBER, stroke_width=1.8,
            )
            med_line = DashedLine(
                [med_x, line_bot, 0], [med_x, line_top, 0],
                dash_length=0.09, stroke_color=C_TEAL, stroke_width=1.8,
            )

            # Labels (mean above, median below top)
            mean_tag = Text("mean", font_size=9, color=C_AMBER)\
                .move_to([mean_x, line_top + 0.18, 0])
            med_tag  = Text("med",  font_size=9, color=C_TEAL)\
                .move_to([med_x,  line_top + 0.18 - 0.30, 0])

            # bars kept separate from grp so FadeIn and GrowFromEdge don't conflict
            grp.add(mean_line, med_line, mean_tag, med_tag)
            all_groups.add(grp)

            self.play(FadeIn(grp), run_time=0.35)
            self.play(
                LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars],
                            lag_ratio=0.08),
                run_time=0.65,
            )
            all_groups.add(*bars)
            self.wait(0.5)

        # Shared legend
        leg_mean = Text("— mean", font_size=11, color=C_AMBER)\
            .move_to([-1.20, BASELINE_Y - 0.45, 0])
        leg_med  = Text("— median", font_size=11, color=C_TEAL)\
            .move_to([ 1.00, BASELINE_Y - 0.45, 0])
        self.play(FadeIn(leg_mean), FadeIn(leg_med), run_time=0.30)

        caption = (
            Text(
                "right-skewed: mean > median  ·  "
                "left-skewed: mean < median  ·  "
                "symmetric: mean ≈ median",
                font_size=11, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )
        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.5)
