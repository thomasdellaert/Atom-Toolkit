# Rewriting the Grotrian renderer on direct SVG

**Status: phases 1-3 implemented** (skeleton, columns, linear + compressed energy scales,
level lines, term-symbol labels, PAVA label de-overlap, themes, `_repr_svg_`, `save()`).
The old matplotlib `Grotrian` and the broken `arrows.py` are deleted. Phases 2–5 below are
still plans.

Phase 3 adds transitions. Notes from that build:

- Arrows resolve endpoints through `Scene.anchors`, never raw energies, so a transition
  into a magnified fan lands on the sublevel line actually drawn. Endpoints walk up the
  hierarchy when a level is not drawn itself, and a transition to a level absent from the
  diagram warns and is skipped rather than guessing.
- A manifold drawn as a fan has an anchor spanning the whole band, whose centre falls
  *between* sublevel lines -- so arrows attach to the band edge facing their partner
  instead, and always meet a drawn line.
- Colour comes from the wavelength (Bruton's spectral fit), fading to grey outside the
  visible band; non-visible lines also dash themselves, since colour alone is not a
  reliable signal. Wavy lines are exempt from auto-dashing -- a dashed squiggle is noise.
- Arc control points are perpendicular to the chord. Offsetting along +x, the obvious
  first implementation, barely bends a diagonal.
- `svgwrite` hands out incrementing auto-ids, which broke the documented purity of
  `render()`; arrowhead marker ids are now derived from the marker's own appearance.

Phase 2 adds hyperfine and Zeeman substructure, drawn *in place* on a single global
magnification. Notes from that build:

- The global scale is never set by the smallest gap (§9.3) -- verified against ¹³⁷Ba⁺ 5d
  ²D₅/₂, whose F=4/F=3 pair is 0.496 MHz apart in a 135 MHz span.
- It is set by the largest **excited-state** splitting, not the largest overall. Measured
  across the bundled species the ground state runs 5.5x (⁴³Ca⁺), 5.4x (¹³⁷Ba⁺), 8.4x
  (⁸⁷Rb) and 7.9x (¹³³Cs) wider than anything else, so scaling to it leaves every other
  manifold a sliver. The ground state overflows proportionally instead, and the plot area
  is inset by the worst fan overhang so nothing runs off the canvas.
- Per-manifold scale overrides exist as §9.9 promised (`hf_scale='fit'`,
  `hf_exaggerate=N`, and `z_` equivalents), and any manifold drawn off the shared scale is
  annotated with its true span. The canvas grows around exaggerated fans rather than
  compressing the energy span.
- Zeeman can be **exaggerated** (`z_scale='auto'`) since at ordinary fields it is
  sub-pixel next to the hyperfine splitting; the vertical offset is decomposed so
  exaggerating Zeeman never rescales hyperfine, and the scale bar reports the factor.
- **Lines stay where the physics puts them; only labels relax.** Near-degenerate sublevels
  are drawn on top of each other, and their `F` labels fan out on leaders. That keeps the
  geometry honest and the figure readable at the same time.
- Zeeman shares the *same* vertical scale as hyperfine rather than getting its own. Zeeman
  sublevels are displaced by the hyperfine shift plus a much smaller Zeeman shift, so a
  separate scale would silently rescale the hyperfine splitting that dominates them. At
  5 G in ¹⁷¹Yb⁺ the Zeeman splitting is correctly invisible next to a 12.6 GHz hyperfine
  splitting.
- Manifolds with no hyperfine data — still the majority — fall back to an evenly spaced
  schematic fan marked with a **dashed** bracket, so "we don't know" never reads as
  "these are the splittings".

Phase 1 notes from the build:

- Label de-overlap was pulled forward from phase 4. Without it a 14-level diagram is
  illegible, so "phase 1 is more useful than today" would not have been true otherwise.
- `Theme.label_gutter` reserves fixed per-column space for labels, which is option (c) from
  §7 and means no font metrics are needed yet. Verified: no label crosses into the next
  column on any bundled species.
- `MultiTerm.__getattribute__` forwards only `Term.__slots__`, so it proxies attributes but
  not methods; `markup_runs`/`to_markup` needed explicit delegation.
- `examples/grotrian_gallery.py` replaces `grotrians.ipynb` as the visual test.

## 1. What is actually wrong today

I surveyed `atomtoolkit/render/grotrian.py` (480 lines) and rendered the demo from
`grotrians.ipynb` to see the output rather than guess at it. The problems fall into three
groups.

### Missing capability

- **There are no labels at all.** Not the term symbol, not F, not m_F. The code says so at
  line 214: `# TODO: Labels (...)`. A Grotrian diagram without labels is close to useless,
  and this is the single biggest gap.
- **No collision handling.** In the 29-level ¹⁷¹Yb⁺ demo, the Zeeman fans at J=2 overlap
  into an unreadable band, and wide fans cross into neighbouring J columns.
- **`arrows.py` does not import.** It ends mid-function at `def WigglyArrow(...)`:
  an `IndentationError`. Nothing references it, so it is simply dead broken code.
- **No tests.** Zero coverage for the whole module.

### Wrong tool

- The output carries a matplotlib **frame, ticks and axes**. The x axis shows J as though
  it were continuous (0…7); the y axis shows unlabelled THz. Neither belongs on a Grotrian
  diagram, and hiding them is extra work per figure.
- **Scale factors are hand-tuned magic numbers.** The demo passes `hf_scale=9000`,
  `z_scale=20`, `fill_factor=0.9`. These exist because hyperfine structure is ~10⁻⁵ of the
  gross spacing, and forcing both onto one linear axis cannot work without a fudge factor.
  This is the core reason the current output never looks right.
- Rich text is effectively impossible. Proper term symbols (⁴f¹⁴6s ²S₁/₂ — superscript
  multiplicity, subscript J, ° for odd parity) need real inline markup. `Term` currently
  only offers ASCII `term_name` like `2S1/2`.

### Structural

- **`render()` is not idempotent.** Verified: rendering twice takes `lines_df` from 67 rows
  to 134. Re-running a notebook cell silently corrupts the figure — in the one environment
  this module is meant for.
- **`level_ordering` accepts only `'J'`.** `calculate_x0` falls off the end and returns
  `None`; setting `level_ordering = 'energy'` fails with
  `TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'`.
- Geometry and drawing are entangled, so layout cannot be tested without matplotlib.
- `pandas` is used as a mutable accumulator holding objects and kwargs dicts in cells, with
  `pd.concat` in a loop and linear scans (`lines_df[lines_df['level'] == level]`) for
  lookups. It is the wrong data structure, though **not currently a speed problem** — 40
  levels render in 0.11 s, so this is a clarity issue, not a performance one.
- `_process_level_kwargs` returns a 12-tuple; ~20 keyword arguments flow through untyped
  dicts with no validation. `squeeze` means three different things depending on whether it
  is `True`, `False`, or a number.

## 2. Why SVG is the right target

A Grotrian diagram is a **diagram**, not a plot. Matplotlib is built around data-space
axes, autoscaling and artists; almost everything above is friction from fighting that.

SVG gives, directly: real inline typography (`<tspan>` with `baseline-shift` handles term
symbols exactly), crisp hairlines at any zoom, precise control of every coordinate,
grouping and reuse (`<g>`, `<defs>`, `<use>`, `<marker>` for arrowheads), and — for free —
tooltips, hover highlighting and click targets. It embeds in notebooks via
`IPython.display.SVG`, in web pages, and drops straight into a paper.

It also removes matplotlib as a dependency of this module.

## 3. Proposed architecture

The most important change is to split the one entangled class into three layers, so that
layout can be tested without rendering anything.

```
   spec  ──────────►  layout  ──────────►  render
 (what to draw)    (where it goes)      (SVG elements)

 GrotrianSpec        LayoutEngine          SvgRenderer
 - LevelSpec         -> Scene              -> str
 - TransitionSpec       (positioned,        (pure, deterministic)
 - Theme                 in diagram units)
```

- **Spec layer** — pure declaration. `add_level(level, detail='hyperfine', ...)` appends an
  immutable `LevelSpec`. No geometry, no physics. Cheap and inspectable.
- **Layout engine** — consumes spec + atom, produces a `Scene`: a flat list of positioned
  primitives (`LevelLine`, `Label`, `Arrow`, `Bracket`) in abstract diagram units. All the
  hard problems live here: column assignment, energy transform, substructure magnification,
  label placement, collision avoidance. **Pure functions over plain dataclasses**, so unit
  tests assert on coordinates.
- **Renderer** — `Scene → str`. Deliberately dumb: no decisions, just element emission.
  Makes golden-file testing meaningful, and leaves room for a second backend later
  (a `Scene → matplotlib` shim would let existing notebooks keep working).

`render()` becomes pure: spec in, string out, callable any number of times.

## 4. Key design decisions

### 4.1 Coordinates

Two independent axes, both pluggable:

- **x is categorical, not continuous.** A `ColumnPolicy` maps each level to a column:
  `by='J'` (today's behaviour), `by='term'`, `by='config'`, `by='parity'`, or an explicit
  callable/dict. The engine then assigns pixel positions with configurable gutters. This
  removes the `calculate_x0` dead end and makes "put the D states in their own column" a
  one-liner.
- **y is an `EnergyScale` object**: `Linear(unit='THz'|'cm-1'|'eV')`, `Log`, or
  `Broken(breaks=[...])` for the common case where the ground state sits far below
  everything else. A `Compressed` variant (piecewise-linear, allocating space by level
  density rather than energy) is what most published diagrams actually use.

### 4.2 Substructure as magnified insets — the fix for `hf_scale`

This is the conceptual fix for the magic numbers, and it is the part of the design with the
most detail behind it, so it gets its own section: see **§9 Inset logic** below.

The short version: hyperfine and Zeeman structure do **not** share the gross energy axis. A
gross level is one line; its hyperfine manifold is drawn in a magnified band around that
line with its own local scale, chosen per manifold and reported rather than invented.

### 4.3 Labels

The missing feature, so it deserves real design.

- **Term markup.** Add `Term.to_markup(fmt)` in `term.py` emitting SVG `<tspan>` or LaTeX
  from the existing quantum numbers. Belongs in `term.py`, not the renderer, because LaTeX
  output is independently useful (papers, matplotlib titles, notebook display).
- **Placement** by a small constraint pass: preferred side (right of the level line by
  default), pushed vertically to resolve overlaps, with a leader line drawn if a label ends
  up more than a threshold from its anchor. Greedy vertical relaxation is enough; this does
  not need a general solver.
- **Level of detail.** Automatically drop m_F labels when adjacent sublevels are closer than
  a legibility threshold, and fall back to labelling the manifold once (`m_F = −2 … +2`).
  Configurable, but the default should be "never illegible".

### 4.4 Transitions

- Replace the 1–9 anchor grid with **intent-based routing**: `route='straight'|'arc'|'step'`
  and automatic port selection (attach to the nearest free point on each level line, biased
  away from crowded regions).
- **Colour by wavelength** properly: implement visible-spectrum wavelength → sRGB, with a
  defined convention outside 380–780 nm (grey, dashed for IR/UV). The `transition_color =
  'spectrum'` attribute already advertises this intent.
- Styles: solid, **wavy** (radiative decay / RF — currently the broken stub), dashed,
  double-headed. As SVG `<marker>` and path definitions in `<defs>`, so they are reusable
  and cheap.
- Optional wavelength labels along the arrow, using SVG `<textPath>`.

### 4.5 Styling and API

Replace ~20 loose kwargs with dataclasses and a resolution order:

```python
Theme(font='...', level_color=..., stroke=..., dark_mode=False)
LevelStyle(color=..., width=..., linewidth=..., label=True, bbox='bracket')
```

resolved as **per-call override > predicate rule > theme default**, with predicate rules
for bulk work:

```python
g.style_levels(lambda l: l.term.J > 1.5, color='steelblue')
```

Target usage — declarative, chainable, idempotent:

```python
from atomtoolkit.render import Grotrian

g = (Grotrian(yb171)
     .add_levels(yb171.levels[:12], detail='hyperfine')
     .add_transition(('4f14.6s 2S1/2', '4f14.6p 2P*1/2'), label='wavelength')
     .style_levels(lambda l: l.term.J > 1.5, color='steelblue'))

g                      # renders inline in Jupyter via _repr_svg_
g.save('yb171.svg')
```

plus a zero-config entry point that must produce something respectable with no tuning:

```python
Grotrian.auto(yb171, n_levels=12).save('yb171.svg')
```

`auto()` is the honest test of whether the defaults are good. If it needs tuning to look
right, the defaults are wrong.

## 5. Testing

Currently zero. The three-layer split is what makes testing tractable:

- **Layout** — pure functions over dataclasses: assert column assignment, that magnified
  bands do not overlap, that labels do not collide, that the energy transform is monotonic.
  Fast, no rendering.
- **Renderer** — golden-file snapshots plus structural assertions (parse the SVG, count
  `<line>` elements, check the `viewBox`, check every level has a label). Assert on
  structure, never on pixels.
- **Regression** — a handful of real species (¹⁷¹Yb⁺ hyperfine, ⁴³Ca⁺ Zeeman, Ho I as the
  pathological case) rendered end to end and checked for "no element outside the viewBox"
  and "no two labels overlapping".

## 6. Suggested phasing

Each phase should leave something usable.

1. **Skeleton + gross structure.** Spec/layout/render split, column policy, linear energy
   scale, level lines, term-symbol labels, `_repr_svg_`, `save()`. Already more useful than
   today, because of labels.
2. **Substructure.** Hyperfine and Zeeman as magnified insets with brackets and automatic
   magnification annotations. Retires `hf_scale`/`z_scale`.
3. **Transitions.** Routing, arrowheads, wavelength colouring, wavy style, labels.
4. **Polish.** Collision avoidance and label relaxation, level-of-detail, broken/compressed
   energy axes, theming and dark mode, `auto()`.
5. **Migration.** Delete the matplotlib `Grotrian` and `arrows.py` outright — no shim, per
   the decision in §8. Update `grotrians.ipynb` to the new API.

## 7. Risks and open questions

- **Text metrics are the main technical risk.** Collision avoidance needs to know how wide a
  label is, and SVG has no layout engine — the browser does that at display time. Options:
  (a) bundle a small advance-width table for one or two standard families and accept
  approximation; (b) use the optional `fonttools` dependency to measure properly;
  (c) sidestep it by reserving fixed label gutters so exact widths never matter. Plan: (c)
  for phase 1, (a) later, (b) only if publication-grade packing demands it. This remains the
  thing most likely to make output look merely "fine" rather than good.
- **Font availability.** Text in an SVG renders with whatever the viewer has. Either stick
  to a websafe stack, or offer `text_as_paths=True` (needs `fonttools`) for publication
  output that is guaranteed identical everywhere.
- **Raster export** (PNG/PDF for journals) needs `cairosvg` or similar. Keep it an optional
  dependency and fail with a clear message.
- **Scope creep into interactivity.** Tooltips and hover are nearly free and very tempting;
  they should still wait until phase 4, and stay optional, since inert SVG is what a paper
  needs.

## 8. Decisions (settled)

1. **The matplotlib `Grotrian` may break.** No back-compatibility shim, no
   `Scene → matplotlib` backend required. `arrows.py` gets deleted.
2. **`svgwrite` is an acceptable dependency.** `fonttools` / `cairosvg` stay optional, for
   text-as-paths and raster export respectively.
3. **Energy axis defaults to linear in energy**, with compressed-by-density available as an
   option.
4. **Phases 1–2 are the first deliverable.**

## 9. Inset logic

### 9.1 Why one fixed scale factor cannot work

I measured the real splittings across three bundled species to check whether one fixed
factor, applied to gross and hyperfine structure alike, could work. It cannot.

| species | manifold | N | span | smallest gap | span/gap |
|---|---|---:|---:|---:|---:|
| ¹⁷¹Yb⁺ | 6s ²S₁/₂ | 2 | 12.643 GHz | 12.643 GHz | 1.00 |
| ¹⁷¹Yb⁺ | 5d ²D₃/₂ | 2 | 8.010 GHz | 8.010 GHz | 1.00 |
| ⁴³Ca⁺ | 4s ²S₁/₂ | 2 | 3.226 GHz | 3.226 GHz | 1.00 |
| ⁴³Ca⁺ | 3d ²D₃/₂ | 4 | 0.567 GHz | 0.139 GHz | 4.07 |
| ⁴³Ca⁺ | 3d ²D₅/₂ | 6 | 0.077 GHz | 0.007 GHz | 11.66 |
| ¹⁶⁵Ho | 4f¹¹6s² ⁴I°₁₅/₂ | 8 | 45.023 GHz | 4.309 GHz | 10.45 |

Two things follow.

**The dynamic range is ~10⁶.** In ⁴³Ca⁺ the median gross gap is 110 THz while the 3d ²D₅/₂
hyperfine span is 0.077 GHz — a ratio of 1.4 million. In the same atom the ground-state
span is 3.2 GHz, forty times larger. No single `hf_scale` serves both; that is precisely
why the current API forces per-call hand tuning, and it is a property of the physics, not
of the old code.

**The `span/gap` column looks like it should set legibility — but must not be used that
way.** It is tempting to size each band so the smallest gap is legible; §9.3 shows why that
fails catastrophically on near-degeneracies. The column is still worth tabulating, because
it says how much internal structure will be visually unresolved at any given scale.

### 9.2 The model

Let the global energy scale be `g` px per Hz. A manifold `m` sits at `Y_m = y_global(E_m)`.
Its sublevels have shifts `δ_i` about the manifold centroid. Then

```
Y_i = Y_m − M_m · ( β·δ_i + (1−β)·schematic_i )
```

- `M_m` [px/Hz] is the **local** scale for that manifold, chosen independently per manifold.
- `β ∈ [0,1]` interpolates realistic (1) ↔ equally-spaced (0). This is the current
  `spacing` keyword, which is worth keeping — it is the one existing knob that expresses
  intent rather than a fudge factor.
- The reported magnification is `M_m / g`.

An inset is therefore just a **local affine map on the energy axis, valid inside a band**.
Everything else — brackets, annotations, fallbacks — is presentation on top of that.

### 9.3 Choosing the magnification — a *global* HF scale by default

**Revised after review.** The original proposal set a per-manifold magnification from the
smallest interval, ``M = min_gap_px / γ``. That is wrong twice over.

**Near-degeneracies blow it up.** ¹³⁷Ba⁺ 5d ²D₅/₂ (A = −12.028, B = 59.520 MHz) has:

| F (energy order) | 4 | 3 | 2 | 1 |
|---|---:|---:|---:|---:|
| shift (MHz) | −30.225 | −29.729 | 33.139 | 104.811 |

F=4 and F=3 sit **0.496 MHz** apart inside a **135 MHz** span — a span/min-gap ratio of
**272**. Demanding 8 px for the smallest gap would require a 2178 px band for one
manifold. And this is not exotic: the same atom's 6p ²P°₃/₂ has a ratio of 19.

**Per-manifold scaling destroys cross-manifold comparison.** If every manifold is
normalised to its own span, a 12 GHz ground-state splitting and a 135 MHz excited-state
one are drawn the same height, which actively misleads.

So the default is a **single global hyperfine magnification** ``M_hf`` [px/Hz] shared by
the whole diagram, driven by the *largest* span present, not the smallest gap:

```
M_hf = theme.hf_band / max(span over all hyperfine manifolds)
```

Consequences, all of them good:

- Relative hyperfine sizes are comparable by eye across the whole figure, which is what a
  reader actually wants to extract.
- Near-degeneracies simply render as near-degenerate. Honest, and no blow-up.
- One number describes the whole diagram, so one scale bar annotates it (§9.6) instead of
  a magnification factor per manifold.

Per-manifold auto-scaling remains available as an opt-in for the case where someone really
wants to resolve one manifold's internal structure, but it is not the default.

### 9.3b Rendering in place, not as a detached inset

**Revised after review.** Insets with leader lines are one convention, but the more common
one — and the better default — is simply to draw the hyperfine structure *exaggerated in
place*: the gross level line is replaced by the fan of its sublevels, centred on where the
gross level sits. No leader, no callout, no separate panel.

The offset/callout modes from §9.4 stay in the ladder as escalations for genuinely
impossible cases, but they are no longer the primary mechanism.

Note that the fan is **not symmetric** about the parent line: the hyperfine centre of
gravity is the degeneracy-weighted mean (Σ(2F+1)·δ_F = 0), not the arithmetic one. Drawing
it centred on the arithmetic mean would be subtly wrong.

Sublevels must also be ordered **by energy, not by F** — a negative A inverts the ordering,
as in ¹³⁷Ba⁺ ²D₅/₂ above, where F runs 4, 3, 2, 1 from the bottom.

### 9.3c Schematic gross structure with quantitative hyperfine

**Added after review.** A genuinely useful combination, and one the architecture supports
for free because the two scales are independent:

- the **gross** axis schematic — ordering and rough relative energy only
  (:class:`CompressedScale`, or a fully schematic scale)
- the **hyperfine** structure to a true, globally consistent scale

That gives a figure where the big picture is legible and the hyperfine splittings can
still be compared and roughly read off. It is the mode most often wanted for a talk or a
paper figure, and it is expressed as::

    Grotrian(atom, scale=CompressedScale(), hf_scale='auto')

### 9.4 Fallback ladder

### 9.5 The dominant edge case: no splitting at all

**Most manifolds have zero hyperfine splitting**, because hyperfine coefficients are only
known (or only entered) for a handful of levels per species. Of the first 14 manifolds:
¹⁷¹Yb⁺ has 7 with `span = 0`, ⁴³Ca⁺ has 9, ¹⁶⁵Ho has 13.

So `Δ = 0` — which would make `M_allowed = h/0` — is not a corner case to guard against, it
is the **common path**. The design must lead with it:

- `Δ = 0` ⇒ realistic spacing is meaningless ⇒ force schematic (`β = 0`) with a fixed pitch.
- The result must be **visually distinguishable** from a real splitting, or the diagram lies
  to the reader. Proposal: schematic-only insets get a dashed bracket and no magnification
  annotation, plus an optional `(no hf data)` note. A reader must never mistake "we don't
  know" for "these are the splittings".
- This also argues for a `Grotrian` warning (or a `.data_gaps()` report) listing manifolds
  drawn schematically for want of data — useful feedback given how many species files carry
  a `Data still needed` block.

### 9.6 Annotation — keeping it honest

Every inset carries a bracket spanning the band on one side, plus one of:

- `annotate='span'` *(default)* — the physical extent, e.g. `12.6 GHz`. Most useful to a
  physicist: it answers "how big is this splitting", not "what zoom did you use".
- `annotate='factor'` — `×2.4·10⁴`, for readers checking the drawing against the axis.
- `annotate='scalebar'` — a short tick marked `1 GHz` inside the band.
- `annotate=None`.

Schematic insets (§9.5) may use **none** of these, since there is no physical extent to
report; they get the dashed bracket instead.

### 9.7 Nesting, and stacked vs fanned sublevels

Zeeman structure nests one level deeper, with the parent being an already-magnified
hyperfine position, so total magnification is `M_m · M_F` relative to global. But Zeeman
structure is laid out differently, and the distinction matters:

- **Hyperfine → stacked.** Full-width lines separated vertically. Purely a y phenomenon.
- **Zeeman → fanned.** Short lines spread *horizontally* by `m_F`, offset vertically by the
  Zeeman shift. At `B = 0` the shifts vanish and the fan is purely horizontal, which is the
  conventional Grotrian look.

Because the fan is horizontal, the Zeeman inset's constraint is on **width**, competing with
the column gutter rather than with vertical neighbours — a different packing problem, worth
keeping separate in the code.

This is also the natural integration point with the Zeeman work already in the library:

```python
g.add_level(lvl, detail='zeeman', B=Q_(5.36, 'G'))
```

which should route through `zlevel.level_hz_at(B)`, and therefore pick up exact
`H_HF + H_Z` diagonalisation for free on any manifold with `use_nonlinear_zeeman = True`.
Drawing a real Breit-Rabi fan, including the curvature near a magic field, then costs
nothing extra.

### 9.8 Anchors — why transitions depend on this

Every drawn object (gross level, hyperfine sublevel, Zeeman sublevel) must register its
**final drawn** anchor box in the `Scene`, keyed by the level object. Transitions resolve
endpoints against those boxes, never against true energies — otherwise an arrow into a
magnified manifold lands nowhere near the line it is supposed to touch.

The current code gets this right in spirit with `lines_df['bbox']`; it is the one part of
the existing design worth carrying over, as `Scene.anchors: dict[BaseLevel, Anchor]`.

Two cases to get right:
- A transition to a manifold that is drawn *collapsed* must fall back to the gross line's
  anchor. This is what today's `override_bbox` keyword gropes at.
- A transition **within** one manifold (a microwave clock transition, F=0 → F=1) lives
  entirely inside one inset and must route inside the band.

### 9.9 Worked example

⁴³Ca⁺, first 14 levels, on a 620 px plot, with ``theme.hf_band = 44`` px. The largest
hyperfine span present is the 4s ²S₁/₂ ground state at 3.226 GHz, so

```
M_hf = 44 px / 3.226 GHz = 1.36e-8 px/Hz
```

and every manifold is drawn with that one factor:

| manifold | span | drawn band | comment |
|---|---:|---:|---|
| 4s ²S₁/₂ | 3.226 GHz | 44.0 px | sets the scale |
| 3d ²D₃/₂ | 0.567 GHz | 7.7 px | visibly smaller, correctly so |
| 3d ²D₅/₂ | 0.077 GHz | 1.0 px | essentially a single line |
| 4p ²P°₃/₂ | 0.370 GHz | 5.0 px | |

²D₅/₂ collapsing to ~1 px is the *right* answer: its hyperfine structure genuinely is 40×
finer than the ground state's, and the figure says so. A reader who needs that manifold
resolved asks for it explicitly, and gets a per-manifold override plus its own annotation.

Compare with the rejected per-manifold rule, which would have drawn all four bands the same
height and implied they were comparable splittings.

### 9.10 Algorithm sketch

```
for each column:
    manifolds ← specs in this column, sorted by Y
    for each manifold m:
        δ ← sublevel shifts                     # from the atom, honest physics
        Δ ← max δ − min δ ;  γ ← min consecutive gap
        if Δ == 0:            mode ← schematic          # §9.5, the common case
        else:                 M_req ← min_gap_px / γ
    resolve vertical packing:                            # 1-D interval problem
        h_available[m] ← gap to neighbours − padding
        M[m] ← min(M_req[m], h_available[m] / Δ[m])
        if M[m] < M_req[m]:  escalate m per the ladder   # §9.4
    emit band geometry, bracket, annotation
    register anchors for every sublevel                  # §9.8
```

The packing step is the only non-trivial part, and it is one-dimensional: intervals with
fixed centres and shrinkable half-heights. A single pass that caps each band at half the
distance to its nearest neighbour handles almost everything; a second relaxation pass can
redistribute slack from manifolds that did not need their full allowance.
