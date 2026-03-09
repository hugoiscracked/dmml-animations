"""
W01 — Data Cleaning

  Phase 1: Missing values.
           An Age column with NaN cells is shown alongside three
           strategies side by side: drop rows, mean imputation,
           and forward-fill — colour-coded for instant comparison.

  Phase 2: Encoding.
           A categorical Sex column is one-hot encoded into two binary
           dummy columns; the dummy-variable trap is noted.

  Phase 3: Normalisation.
           Original Age values are mapped by min-max scaling (→ [0, 1])
           and standardisation (→ mean=0, σ=1) on three parallel number
           lines, showing that shape is preserved while scale changes.

Render:
  ../../env/bin/manim -pql data_cleaning.py DataCleaning
  ../../env/bin/manim -pqh data_cleaning.py DataCleaning
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


def _cell(text, cx, cy, w, h, bg=None, tc=C_DIM, font_size=10, bold=False):
    rect = Rectangle(
        width=w, height=h,
        fill_color=bg or C_BG, fill_opacity=0.22 if bg else 0.0,
        stroke_color=C_DEAD, stroke_width=0.8,
    ).move_to([cx, cy, 0])
    kw = {"weight": BOLD} if bold else {}
    txt = Text(text, font_size=font_size, color=tc, **kw).move_to([cx, cy, 0])
    return VGroup(rect, txt)


class DataCleaning(Scene):

    def construct(self):
        self.camera.background_color = C_BG
        self._phase1()
        self._phase2()
        self._phase3()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Missing values
    # ══════════════════════════════════════════════════════════════════════════

    def _phase1(self):
        title = (
            Text("Data Cleaning", font_size=24, color=C_WHITE, weight=BOLD)
            .to_edge(UP, buff=0.25)
        )
        sub = (
            Text("Missing values — three strategies compared side by side",
                 font_size=13, color=C_DIM)
            .next_to(title, DOWN, buff=0.12)
        )
        self.play(FadeIn(title), FadeIn(sub), run_time=0.5)

        # ── Layout ────────────────────────────────────────────────────────────
        COL_XS     = [-4.5, -1.5, 1.5, 4.5]
        COL_NAMES  = ["Original", "Drop rows", "Mean impute", "Forward-fill"]
        COL_COLORS = [C_DIM,      C_RED,       C_AMBER,       C_TEAL      ]
        CW, CH     = 1.50, 0.42
        Y0         = 1.40   # header row centre

        # Age values for each column strategy
        # (val, text_color, bg_color_or_None)
        def orig_cells():
            return [
                ("22",  C_WHITE, None   ),
                ("NaN", C_RED,   C_RED  ),
                ("35",  C_WHITE, None   ),
                ("NaN", C_RED,   C_RED  ),
                ("28",  C_WHITE, None   ),
            ]

        def drop_cells():
            return [
                ("22",  C_WHITE, None  ),
                ("—",   C_DEAD,  None  ),
                ("35",  C_WHITE, None  ),
                ("—",   C_DEAD,  None  ),
                ("28",  C_WHITE, None  ),
            ]

        def mean_cells():
            return [
                ("22",   C_WHITE, None   ),
                ("28.3", C_AMBER, C_AMBER),
                ("35",   C_WHITE, None   ),
                ("28.3", C_AMBER, C_AMBER),
                ("28",   C_WHITE, None   ),
            ]

        def ffill_cells():
            return [
                ("22", C_WHITE, None  ),
                ("22", C_TEAL,  C_TEAL),
                ("35", C_WHITE, None  ),
                ("35", C_TEAL,  C_TEAL),
                ("28", C_WHITE, None  ),
            ]

        all_cell_data = [orig_cells(), drop_cells(), mean_cells(), ffill_cells()]

        # Strategy name labels (above header row)
        strat_lbls = VGroup(*[
            Text(name, font_size=11, color=col, weight=BOLD)
            .move_to([COL_XS[c], Y0 + CH + 0.22, 0])
            for c, (name, col) in enumerate(zip(COL_NAMES, COL_COLORS))
        ])

        # "Age" header cells
        headers = VGroup(*[
            _cell("Age", COL_XS[c], Y0, CW, CH,
                  bg=C_DEAD, tc=col, font_size=11, bold=True)
            for c, col in enumerate(COL_COLORS)
        ])

        # Data cell groups (one VGroup per column)
        col_groups = []
        for c, cells in enumerate(all_cell_data):
            grp = VGroup(*[
                _cell(val, COL_XS[c], Y0 - (r + 1) * CH, CW, CH,
                      bg=bg, tc=tc, font_size=11)
                for r, (val, tc, bg) in enumerate(cells)
            ])
            col_groups.append(grp)

        # Annotations below table
        ANN_Y = Y0 - 6 * CH - 0.55
        annotations = [
            None,
            Text("rows removed\n→ smaller dataset", font_size=9, color=C_RED)
            .move_to([COL_XS[1], ANN_Y, 0]),
            Text("NaN → column mean\n→ preserves size", font_size=9, color=C_AMBER)
            .move_to([COL_XS[2], ANN_Y, 0]),
            Text("NaN → previous value\n→ assumes ordering", font_size=9, color=C_TEAL)
            .move_to([COL_XS[3], ANN_Y, 0]),
        ]

        # ── Animate ───────────────────────────────────────────────────────────
        self.play(FadeIn(strat_lbls), FadeIn(headers), run_time=0.40)
        self.play(FadeIn(col_groups[0]), run_time=0.35)
        self.wait(0.4)

        for c in range(1, 4):
            anims = [FadeIn(col_groups[c])]
            if annotations[c]:
                anims.append(FadeIn(annotations[c]))
            self.play(*anims, run_time=0.40)
            self.wait(0.75)

        all_anns = VGroup(*[a for a in annotations if a is not None])
        self._title  = title
        self._p1_all = VGroup(sub, strat_lbls, headers, *col_groups, all_anns)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Encoding
    # ══════════════════════════════════════════════════════════════════════════

    def _phase2(self):
        self.play(FadeOut(self._p1_all), run_time=0.40)

        sub = (
            Text("Encoding — convert categories to numbers the model can use",
                 font_size=13, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        # ── Data ──────────────────────────────────────────────────────────────
        SEX_VALS  = ["male", "female", "female", "male", "male"]
        MALE_OHE  = [1, 0, 0, 1, 1]
        FEM_OHE   = [0, 1, 1, 0, 0]

        CW, CH = 1.50, 0.40
        Y0     = 1.20   # header row centre
        NR     = len(SEX_VALS)

        X_SEX   = -4.5
        X_MALE  = -1.5
        X_FEM   =  0.5

        # Original Sex column
        sex_header = _cell("Sex", X_SEX, Y0, CW, CH,
                           bg=C_DEAD, tc=C_DIM, font_size=11, bold=True)
        sex_data = VGroup(*[
            _cell(val, X_SEX, Y0 - (r + 1) * CH, CW, CH,
                  bg=C_TEAL if val == "male" else C_PURP,
                  tc=C_TEAL if val == "male" else C_PURP,
                  font_size=11)
            for r, val in enumerate(SEX_VALS)
        ])

        # One-hot columns
        male_header = _cell("Sex_male", X_MALE, Y0, CW, CH,
                            bg=C_DEAD, tc=C_TEAL, font_size=11, bold=True)
        male_data = VGroup(*[
            _cell(str(v), X_MALE, Y0 - (r + 1) * CH, CW, CH,
                  bg=C_TEAL if v == 1 else None,
                  tc=C_TEAL if v == 1 else C_DEAD,
                  font_size=12, bold=(v == 1))
            for r, v in enumerate(MALE_OHE)
        ])

        fem_header = _cell("Sex_female", X_FEM, Y0, 1.70, CH,
                           bg=C_DEAD, tc=C_PURP, font_size=11, bold=True)
        fem_data = VGroup(*[
            _cell(str(v), X_FEM, Y0 - (r + 1) * CH, 1.70, CH,
                  bg=C_PURP if v == 1 else None,
                  tc=C_PURP if v == 1 else C_DEAD,
                  font_size=12, bold=(v == 1))
            for r, v in enumerate(FEM_OHE)
        ])

        # Arrow and label
        arr_y  = Y0 - NR * CH / 2   # vertical centre of table
        arr    = Arrow(
            [X_SEX + CW / 2 + 0.10, arr_y, 0],
            [X_MALE - CW / 2 - 0.10, arr_y, 0],
            buff=0.0, color=C_DIM,
            stroke_width=1.6, max_tip_length_to_length_ratio=0.20,
        )
        arr_lbl = (
            Text("one-hot", font_size=9, color=C_DIM)
            .next_to(arr, UP, buff=0.06)
        )

        # Note box (right side)
        NOTE_X = 3.30
        note_bg = RoundedRectangle(
            width=2.60, height=1.40, corner_radius=0.10,
            fill_color=C_DEAD, fill_opacity=0.20,
            stroke_color=C_DEAD, stroke_width=1.0,
        ).move_to([NOTE_X, arr_y, 0])
        note_t1 = Text("k = 2 categories",    font_size=10, color=C_WHITE).move_to([NOTE_X, arr_y + 0.36, 0])
        note_t2 = Text("→ 2 dummy columns",   font_size=10, color=C_DIM  ).move_to([NOTE_X, arr_y + 0.08, 0])
        note_t3 = Text("drop one to avoid",   font_size=9,  color=C_AMBER).move_to([NOTE_X, arr_y - 0.22, 0])
        note_t4 = Text("multicollinearity",   font_size=9,  color=C_AMBER).move_to([NOTE_X, arr_y - 0.46, 0])
        note = VGroup(note_bg, note_t1, note_t2, note_t3, note_t4)

        caption = (
            Text(
                "Sex_female = 1 − Sex_male  ·  one column is redundant"
                "  ·  keep k−1 dummies",
                font_size=11, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )

        # ── Animate ───────────────────────────────────────────────────────────
        self.play(FadeIn(sex_header), FadeIn(sex_data), run_time=0.40)
        self.wait(0.3)
        self.play(GrowArrow(arr), FadeIn(arr_lbl), run_time=0.40)
        self.play(FadeIn(male_header), FadeIn(male_data), run_time=0.40)
        self.play(FadeIn(fem_header),  FadeIn(fem_data),  run_time=0.40)
        self.play(FadeIn(note), run_time=0.35)
        self.play(FadeIn(caption), run_time=0.30)
        self.wait(1.8)

        self._p2_all = VGroup(
            sub, sex_header, sex_data,
            arr, arr_lbl,
            male_header, male_data,
            fem_header, fem_data,
            note, caption,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Normalisation
    # ══════════════════════════════════════════════════════════════════════════

    def _phase3(self):
        self.play(FadeOut(self._p2_all), run_time=0.40)

        sub = (
            Text("Normalisation — rescale features so no column dominates",
                 font_size=13, color=C_DIM)
            .next_to(self._title, DOWN, buff=0.12)
        )
        self.play(FadeIn(sub), run_time=0.28)

        # ── Data ──────────────────────────────────────────────────────────────
        ages    = np.array([22, 28, 35, 38, 45, 52], dtype=float)
        mm      = (ages - ages.min()) / (ages.max() - ages.min())   # min-max
        z       = (ages - ages.mean()) / ages.std()                  # z-score

        AX_L, AX_R = -3.20, 3.20    # axis x extent
        AX_SPAN    = AX_R - AX_L
        Y_LINES    = [1.30, 0.00, -1.30]
        DOT_R      = 0.09
        DOT_COL    = [C_BLUE, C_GREEN, C_AMBER]

        def to_x(vals, v_lo, v_hi):
            return AX_L + (vals - v_lo) / (v_hi - v_lo) * AX_SPAN

        # Original: pad range slightly so edge dots don't sit on endpoints
        orig_xs  = to_x(ages, 18.0, 56.0)
        mm_xs    = to_x(mm,    0.0,  1.0)
        z_xs     = to_x(z,    -2.3,  2.3)

        configs = [
            # (dot_xs, label, formula_line1, formula_line2, tick_xs, tick_lbls)
            (orig_xs,
             "Original",
             "range: 22 – 52",
             "(no rescaling)",
             to_x(np.array([20., 30., 40., 50.]), 18., 56.),
             ["20", "30", "40", "50"]),
            (mm_xs,
             "Min-Max",
             "x' = (x − min) / (max − min)",
             "maps to [0, 1]",
             to_x(np.array([0., 0.5, 1.0]), 0., 1.),
             ["0", "0.5", "1"]),
            (z_xs,
             "Standard",
             "x' = (x − μ) / σ",
             "μ = 36.7,  σ = 10.2",
             to_x(np.array([-2., -1., 0., 1., 2.]), -2.3, 2.3),
             ["-2", "-1", "0", "1", "2"]),
        ]

        all_frames = VGroup()
        all_dot_groups = []

        for (dot_xs, label, form1, form2, tick_xs, tick_lbls), ly, dcol in \
                zip(configs, Y_LINES, DOT_COL):

            frame = VGroup()

            # Axis
            frame.add(Line([AX_L, ly, 0], [AX_R, ly, 0],
                           stroke_color=C_DIM, stroke_width=1.2))

            # Tick marks + labels
            for tx, tlbl in zip(tick_xs, tick_lbls):
                frame.add(Line([tx, ly - 0.09, 0], [tx, ly + 0.09, 0],
                               stroke_color=C_DIM, stroke_width=0.8))
                frame.add(Text(tlbl, font_size=8, color=C_DIM)
                          .move_to([tx, ly - 0.24, 0]))

            # Left label
            frame.add(
                Text(label, font_size=12, color=C_WHITE, weight=BOLD)
                .move_to([AX_L - 1.30, ly, 0])
            )

            # Right annotation (two lines)
            frame.add(
                Text(form1, font_size=9, color=dcol)
                .move_to([AX_R + 1.55, ly + 0.17, 0])
            )
            frame.add(
                Text(form2, font_size=9, color=C_DIM)
                .move_to([AX_R + 1.55, ly - 0.17, 0])
            )

            all_frames.add(frame)

            # Dots (animated separately)
            dots = VGroup(*[
                Dot([dx, ly, 0], radius=DOT_R,
                    fill_color=dcol, fill_opacity=0.90)
                for dx in dot_xs
            ])
            all_dot_groups.append(dots)

        # Caption
        caption = (
            Text(
                "shape is preserved  ·  "
                "use Standard for distance-based models (kNN, SVM, PCA)",
                font_size=11, color=C_DIM,
            )
            .to_edge(DOWN, buff=0.32)
        )

        # ── Animate ───────────────────────────────────────────────────────────
        for frame, dots in zip(all_frames, all_dot_groups):
            self.play(FadeIn(frame), run_time=0.35)
            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.12),
                run_time=0.55,
            )
            self.wait(0.4)

        self.play(FadeIn(caption), run_time=0.35)
        self.wait(2.5)
