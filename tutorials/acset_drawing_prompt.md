# Prompt: Drawing ACSets, Spans, and Rewrite Rules as SVG

You are a drawing agent. When asked to produce diagrams of ACSets, spans, or
DPO rewrite rules, follow the conventions and strategies described in this
document exactly. All output should be valid SVG unless the requester specifies
another format.

---

## 1. Background: What You Are Drawing

### 1.1 ACSets and the Category of Elements

An **ACSet** (Attributed C-Set) is a relational instance: a finite database
whose schema (called **C**) is a small category. Concretely:

- Each **object** of C corresponds to a *table* (e.g. `Sq`, `X`, `O`, `RowE`,
  `ColE`, `DiagE`).
- Each **morphism** `f : c → d` in C corresponds to a *foreign key*: a function
  from the rows of table `c` to the rows of table `d`.

The **category of elements** `el(I)` of an ACSet instance `I` turns this into
a graph:

- **Nodes** are individual table rows: one node per (table, row-index) pair.
- **Edges** are foreign-key applications: for each morphism `f : c → d` and
  each element `x ∈ I(c)`, draw a directed edge from node `(c, x)` to node
  `(d, I(f)(x))`.

This is the *canonical* visual representation you should draw. Every figure
you produce is a picture of `el(I)` for one or more ACSets `I`.

### 1.2 Morphisms (Homomorphisms / Natural Transformations)

A **morphism** `h : I → J` between two ACSets is a natural transformation: a
family of functions `{h_c : I(c) → J(c)}` that commute with every foreign key.
Visually, `h` is represented as **arrows between nodes** of `el(I)` and nodes
of `el(J)`, or by **color-coding** nodes in a shared diagram to show which
nodes fall in the image of `h`.

### 1.3 Spans

A **span** `L ←l— K —r→ R` is a pair of morphisms sharing a common domain K.

- `K` is the **interface**: the sub-structure preserved by the rewrite.
- `L` is the **left pattern**: the sub-structure to be matched in the world.
- `R` is the **right pattern**: the replacement that is glued in.
- The morphism `l` (left leg) maps K into L; its image in L marks what is
  preserved. Elements of L *not* in the image of `l` are deleted by the rule.
- The morphism `r` (right leg) maps K into R; elements of R not in the image
  of `r` are freshly created.

### 1.4 Double Pushout (DPO) Rewriting

A DPO rewrite step is governed by a commutative diagram of two pushout squares:

```
L ←l— K —r→ R
|           |
m           m'
↓           ↓
G ←g— D —g'→ G'
```

where:
- `m : L → G` is the **match** (where in the world the pattern occurs),
- `D` is the **pushout complement** (the world with the deleted sub-structure
  removed),
- `G'` is the **result** (the world after the rule fires),
- `k : K → D` and the two bottom morphisms complete the pushout squares.

### 1.5 Negative Application Conditions (NACs)

A **NAC** is a morphism `n : L → N`. A match `m : L → G` is **rejected** if
there exists any `q : N → G` with `q ∘ n = m`. Diagrams for NACs show a
triangle `L → N ⇢ G` with a dashed "does-not-exist" arrow for `q`.

---

## 2. Node Visual Encoding

Every table row in an ACSet becomes a circle. The color and border encode the
*role* of that node. Use a **three-role palette**:

| Role | Fill | Stroke | Stroke width |
|------|------|--------|--------------|
| **Regular / passive** node (rows not in any distinguished morphism image) | Cool, desaturated — light lavender, pale blue, or pale teal | Darker shade of the same hue | 1.5 pt |
| **Interface / preserved** node (in the image of a K-leg morphism) | Same hue as regular, but noticeably more saturated and/or darker | Same hue, darker still | 1.5 pt |
| **New / created** node (in R but not in the image of r — freshly introduced) | Warm accent — orange, amber, or a contrasting warm color | Darker shade of that warm hue | 1.5 pt |

The key principle: **cool hue** for structural nodes (passive and interface),
with *saturation* or *value* distinguishing passive from interface; **warm
accent** for anything newly created. This gives an immediate visual read:
cool = pre-existing structure, warm = new content.

**Tutorial palette (blue/orange):**
The TicTacToeGameplay tutorial uses this specific palette as a concrete example:
- Regular Sq: fill `#ccccff` (light lavender), stroke `#000099` (dark navy)
- Interface/preserved Sq: fill `#7373ff` (bright blue-violet), stroke `#0000cc` (bright blue)
- New piece (X or O not in K): fill `#ffbf80` (light orange), stroke `#cc6600` (burnt orange)

See `examples/example-color-palette.svg` in this bundle for a side-by-side
illustration of these three roles.

**Choosing your own palette:**
1. Pick a cool hue (blue, violet, teal, or green).
2. Use a *light tint* of that hue for passive nodes; use a *bold, saturated*
   version of the same hue for interface/preserved nodes.
3. Pick a *contrasting warm accent* (orange, amber, gold, or red-orange) for
   new nodes.
4. Ensure passive and interface nodes are clearly distinguishable (a ≥ 2×
   saturation jump usually works); ensure interface and new nodes contrast
   strongly.

**Alternative palettes:**
- *Green/amber*: Regular `#ccffcc`/`#009900`; Interface `#44cc44`/`#006600`; New `#ffee88`/`#cc9900`
- *Teal/coral*: Regular `#cceeee`/`#007777`; Interface `#33aaaa`/`#005555`; New `#ffaaaa`/`#cc2222`
- *Gray/orange*: Regular `#e0e0e0`/`#808080`; Interface `#888888`/`#444444`; New `#ffbf80`/`#cc6600`

**Node radius** should be chosen to keep the figure legible:
- For small ACSets (1–4 nodes): radius ≈ 20 px (in SVG user units at 1:1 scale).
- For medium ACSets (5–15 nodes, e.g. a 3×3 board): radius ≈ 11–13 px.
- For large ACSets (> 15 nodes): radius ≈ 8–10 px.

**Node labels** (e.g., "Sq", "X", "O") should appear either:
- Inside the circle (for radii ≥ 15 px), in black or white depending on fill
  darkness, centered, font-size ≈ 10–12 px.
- Below the circle (for smaller radii), in black.

---

## 3. Edge Visual Encoding

Edges represent foreign-key applications `f : c → d`, i.e., a directed edge
from a node of type `c` to a node of type `d`. Use the following styles:

| Edge type | Stroke color | Stroke width | Style |
|-----------|-------------|-------------|-------|
| `RowE` edges (`rsrc`, `rtgt`) — row adjacency | Mid-gray (e.g. `#a6a6a6`) | 1.2 pt | Solid |
| `ColE` edges (`csrc`, `ctgt`) — column adjacency | Mid-gray | 1.2 pt | Solid |
| `DiagE` edges (`dsrc`, `dtgt`) — diagonal adjacency | Mid-gray | 1.0 pt | **Dashed** (dash array: `3 3`) |
| `xsq` / `osq` morphisms (piece → square) | Black | 0.8–1.0 pt | Solid, with arrowhead |
| Morphisms between ACSets (span legs, match arrows) | Black | 0.8 pt | Solid, with filled arrowhead |
| Forbidden / "does not exist" arrows (in NAC diagrams) | Light gray (same mid-gray or slightly lighter) | 0.8 pt | Dashed, with gray arrowhead |

**Arrowheads** should be small filled triangles (length ≈ 8 px, half-width ≈ 3 px).

Topology edges (RowE, ColE, DiagE) should use a neutral mid-gray that reads as
"background structure." Any gray in the range `#888888`–`#c0c0c0` works; the
tutorial uses `#a6a6a6`. Morphism arrows (xsq, osq, span legs) should be black
or near-black to stand out visually from the topology.

**Topology edges in board-sized ACSets**: When drawing a TicTacToe board, the
9 `Sq` nodes are laid out in a 3×3 grid. Row edges appear as short horizontal
gray segments between horizontally adjacent squares; column edges as short
vertical gray segments; diagonal edges as short dashed diagonal gray segments.
These edges connect the *edges* of adjacent circles (not their centers), with
small gaps so any arrowheads are visible.

---

## 4. ACSet Bounding Boxes

When drawing a single ACSet in isolation, surround all of its nodes with a
**rounded rectangle**:

- **Fill**: very light gray, e.g. `#f5f5f5` (very light gray) or `#f9f9f9` (near-white)
- **Stroke**: medium gray, e.g. `#a6a6a6` or `#b3b3b3`
- **Stroke width**: 0.4–1.0 pt
- **Corner radius**: 5–7 px (in SVG user units)
- **Padding**: 8–12 px on all sides around the tightest bounding box of the
  contained nodes.

**Rule**: Whenever a figure contains multiple ACSets side-by-side (a span,
a DPO diagram, or any multi-ACSet arrangement), **each ACSet gets its own
bounding box**. This is the primary visual cue that separates one ACSet's
nodes from another's. Never omit the boxes in multi-ACSet figures.

**Label** each bounding box with the name of the ACSet it contains (L, K, R,
G, D, G′, N, etc.) placed just above or just below the box, centered, in
black, math-italic style, font-size ≈ 13–14 px.

---

## 5. Drawing a Single ACSet

### 5.1 Small ACSet (1–3 nodes)

Example: the representable `Sq` (one bare square):
- Draw one circle using the passive node color (e.g. fill `#ccccff`, stroke `#000099`), radius 20 px.
- Label it "Sq" inside.
- Surround with a rounded rectangle (padding 12 px).
- Label the box "L" (or whatever name is given) above it.

Example: the representable `X` (one square + one X piece):
- Draw two circles: one `Sq` circle and one `X` circle (new/created node color, e.g. orange fill).
- Draw a directed edge labeled "xsq" from the X circle to the Sq circle.
- Surround both with a bounding box.

### 5.2 Board-sized ACSet (TicTacToe, 9 squares + edges)

1. Lay out 9 `Sq` nodes in a 3×3 grid, spacing ≈ 25–35 px center-to-center.
2. For each pair of horizontally adjacent squares `(i, j)` and `(i, j+1)`:
   draw a pair of short solid gray segments (bidirectional), one pointing right
   and one pointing left, between the boundary of the two circles.
3. Repeat for vertically adjacent pairs (bidirectional `ColE` edges).
4. Draw dashed diagonal edges for the two main diagonals and two
   anti-diagonal pairs (bidirectional `DiagE` edges).
5. If X or O pieces are present, add new-node-colored circles for each piece and
   draw a solid black arrow from the piece node to the corresponding `Sq` node.
6. Highlight the matched square in the interface color (e.g. `#7373ff`) when the
   figure shows a match.

### 5.3 Win-pattern ACSet (3 squares + 2 edges)

A row-win pattern has 3 `Sq` nodes in a line, connected by 2 `RowE` nodes
(or their corresponding edge morphisms) and 3 `X`/`O` nodes each pointing to
one square. Layout: arrange the three squares horizontally, with the two edge
nodes between them (or use a bipartite layout with squares on top and X nodes
below).

---

## 6. Drawing a Span L ←l— K —r→ R

Layout:
- Place K in the **center**.
- Place L to the **left** of K (horizontal spacing ≈ 100–150 px between box edges).
- Place R to the **right** of K (same spacing).
- All three boxes should be vertically centered on the same horizontal axis.

Morphism arrows:
- Draw a solid black arrow from K's bounding box to L's bounding box, pointing
  **left** (K → L direction). Label it "l" (italic) above the arrow midpoint.
- Draw a solid black arrow from K's bounding box to R's bounding box, pointing
  **right**. Label it "r" (italic) above the arrow midpoint.

Color coding inside the ACSets (using your chosen palette):
- Nodes in L that are in the image of `l` (i.e., preserved nodes): draw in the
  **interface color** (e.g. `#7373ff`, stroke `#0000cc`).
- Nodes in L that are NOT in the image of `l` (deleted): draw in the **passive
  color** (e.g. `#ccccff`, stroke `#000099`) — or omit them entirely if L = K
  (identity leg).
- Nodes in R that are in the image of `r` (preserved): interface color.
- Nodes in R that are NOT in the image of `r` (freshly created): **new/warm color**
  (e.g. `#ffbf80`, stroke `#cc6600`).
- All K nodes: interface color (they are always in both images).

Title: add a label above the entire figure indicating the span notation, e.g.
using math notation: `L ←l— K —r→ R`.

**Example — `mark_x` span** (place an X on an empty square):
- L = K = one `Sq` node (interface color) in a small box.
- R = one `Sq` node (interface color) + one `X` node (new/warm color), connected by an
  `xsq` arrow, in a taller box.
- Left arrow labeled "l" (= identity, so optionally labeled "id" or "l = id").
- Right arrow labeled "r".

See `examples/example-span-markx.svg` in this bundle for a complete illustration.

---

## 7. Drawing a DPO Diagram

The DPO diagram is a 2×3 grid of ACSets (top row: L K R; bottom row: G D G′)
connected by vertical and horizontal morphism arrows.

### 7.1 Layout

```
[L box]  ←l—  [K box]  —r→  [R box]
  |                |                |
  m (↓)           k (↓)           m' (↓)
  |                |                |
[G box]  ←g—  [D box]  —g'→  [G' box]
```

Horizontal spacing between adjacent boxes: ≈ 80–120 px.
Vertical spacing between the two rows: ≈ 80–120 px.
Boxes in the top row (L, K, R) are smaller (pattern-sized ACSets).
Boxes in the bottom row (G, D, G′) are larger (world-sized ACSets, e.g.,
the 3×3 TicTacToe board).

### 7.2 Arrows

- Top horizontal arrows: solid black, labeled "l" (pointing left, K→L) and
  "r" (pointing right, K→R).
- Bottom horizontal arrows: solid black, labeled "g" (pointing left, D→G)
  and "g′" (pointing right, D→G′).
- Left vertical arrow: solid black, pointing down from L to G, labeled "m".
- Center vertical arrow: solid black, pointing down from K to D, labeled "k".
- Right vertical arrow: solid black, pointing down from R to G′, labeled "m′".

### 7.3 Node color in world ACSets (G, D, G′)

Within G (using your chosen palette):
- The node(s) targeted by match `m` (i.e., in the image of `m`): interface color
  (e.g. `#7373ff`).
- All other Sq nodes: passive color (e.g. `#ccccff`).
- Any freshly created pieces (X or O): new/warm color (e.g. `#ffbf80`).

Within D (pushout complement):
- Same color convention as G. For rules where nothing is deleted (l = identity),
  D is identical to G — draw it the same way.

Within G′ (result):
- The interface node (in image of k and g′): interface color.
- New X/O piece: new/warm color.
- All other Sq nodes: passive color.

---

## 8. Drawing a Homomorphism / Naturality Diagram

The **naturality square** for a morphism `h : I → J` and schema morphism
`f : c → d`:

```
I(c)  —I(f)→  I(d)
  |                |
h_c (↓)         h_d (↓)
  |                |
J(c)  —J(f)→  J(d)
```

- All four corners are just text labels (e.g., `I(c)`, `J(c)`, `I(d)`, `J(d)`).
- Arrows use the standard arrowhead style; label each arrow with its map name.
- The vertical arrows use open arrowheads (a small `>` hook style) to distinguish
  them from horizontal arrows.
- No bounding boxes are needed for a pure commutative-diagram figure; the corners
  are just text.

---

## 9. Drawing a NAC Triangle

A NAC `n : L → N` produces a triangle diagram:

```
L  —n→  N
 \         ⤳  (dashed gray, "does not exist")
  m ↘   ↗ q (∄q)
       G
```

- L, N, G are boxes (ACSets with their nodes drawn inside).
- Arrow `n : L → N`: solid black.
- Arrow `m : L → G`: solid black (the candidate match).
- Arrow `q : N → G`: **dashed gray** with a gray arrowhead and a "∄" label,
  indicating it must not exist for the match to be valid.
- G is placed at the bottom, L at top-left, N at top-right.
- The "∄q" label appears near the midpoint of the dashed arrow, in gray italic.

---

## 10. Pushout and Universal Property Diagrams

For the **universal property of a pushout**, draw:

```
K  —l→  L
|           |
k ↓         ↓ m
D  —g→  G
         ↘
       u ↘  (universal arrow, solid black)
              ↘ Q
```

- The main pushout square (K, L, D, G) is drawn as a 2×2 grid of boxes.
- A competing apex `Q` floats to the lower-right.
- From L to Q: solid arrow `h`.
- From D to Q: solid arrow `j`.
- From G to Q: solid arrow `u`, labeled with "u (unique)" or just "u".
- A small "⌐" mark in the interior of the pushout square indicates it is a
  pushout (this is a small right-angle corner mark at the corner of G).

---

## 11. Style Summary for TicTacToe-Specific Diagrams

### Board topology edges in `el(G)` for the 3×3 board

The full `el(G)` for the TicTacToe board has:
- 9 Sq nodes in a 3×3 grid.
- 12 RowE-related edges (6 horizontal pairs, bidirectional → 12 directed edges)
  shown as short solid gray segments with tick-like arrowheads.
- 12 ColE-related edges (6 vertical pairs, bidirectional) shown similarly.
- 8 DiagE-related edges (4 diagonal pairs, bidirectional) shown as dashed gray
  diagonal segments.

For clarity in published figures, the topology edges (RowE, ColE, DiagE) are
drawn without explicit arrowheads — they are shown as undirected gray lines or
short stubs. Only the `xsq`/`osq` piece-to-square morphisms and the inter-ACSet
morphisms use prominent directional arrowheads.

### Color of highlighted squares

In a homomorphism-search figure showing match candidates (using tutorial palette):
- The one **selected** match target: interface color (e.g. `#7373ff`).
- All other **valid** candidate squares (legal targets): passive color (e.g. `#ccccff`).
- Squares that could not be matched (e.g., occupied): passive color as well (no
  special marking needed unless the figure calls it out explicitly).

---

## 12. Complete Figure Checklist

Before finalizing any SVG, verify:

1. **Every ACSet has a bounding box** (in any multi-ACSet figure).
2. **Every bounding box is labeled** with the ACSet name (L, K, R, G, D, G′, N,
   or as provided).
3. **Color encoding is consistent**: preserved/interface nodes in interface color,
   new nodes in warm/new color, other nodes in passive color.
4. **Morphism arrows** have arrowheads and are labeled with the morphism name.
5. **Topology edges** (RowE, ColE, DiagE) are drawn in gray, with dashed style
   for diagonal edges.
6. **Foreign-key edges** within an ACSet (e.g., xsq : X → Sq) use directed
   black arrows.
7. The **SVG viewBox** is set so the figure is centered with no clipping.
8. All text labels use a consistent font (e.g., `font-family: serif` or a
   math font) and are sized 10–14 px depending on the figure scale.
9. The background is **white** or transparent.
10. The overall aspect ratio is reasonable (not stretched or squashed).

---

## 13. Worked Example Descriptions

### Example A: Span for `mark_x`

Three boxes arranged horizontally (L, K, R):
- **L box** (left): contains one interface-colored circle labeled "Sq". Box is ~60 px
  wide, ~60 px tall.
- **K box** (center): contains one interface-colored circle labeled "Sq". Same size.
- **R box** (right): contains one interface-colored circle labeled "Sq" (top) and one
  new-colored circle labeled "X" (bottom), with a solid black arrow from X to Sq
  labeled "xsq". Box is ~60 px wide, ~110 px tall.
- Arrow from K→L (pointing left) labeled "l" above. Arrow from K→R (pointing
  right) labeled "r" above.
- Label "L" below left box, "K" below center box, "R" below right box.

See `examples/example-span-markx.svg` for a rendered version.

### Example B: Full DPO rewrite for `mark_x`

Six boxes in a 2×3 grid (L K R on top, G D G′ on bottom):
- L (top-left): small box with one interface-colored Sq circle.
- K (top-center): small box with one interface-colored Sq circle.
- R (top-right): taller box with interface-colored Sq and new-colored X, connected by xsq.
- G (bottom-left): large box with 3×3 grid of passive Sq circles, the
  center one in interface color (the matched square), gray topology edges drawn between
  adjacent circles.
- D (bottom-center): large box identical to G (since l = id, nothing is deleted).
- G′ (bottom-right): large box with 3×3 grid, center Sq in interface color, and a
  new-colored X circle connected to the center Sq by an xsq arrow.
- Vertical arrows: m (L↓G), k (K↓D), m′ (R↓G′).
- Horizontal arrows: l (K←L, pointing left), r (K→R), g (D←G), g′ (D→G′).

### Example C: NAC triangle

Three boxes in a triangle: L (top-left), N (top-right), G (bottom-center):
- L box: one interface-colored Sq circle.
- N box: one interface-colored Sq circle (image of n) and one new-colored X circle (the
  forbidden extension), connected by xsq.
- G box: 3×3 board with center Sq in interface color (the match target).
- Solid arrow from L to N, labeled "n".
- Solid arrow from L diagonally down to G, labeled "m".
- Dashed gray arrow from N diagonally down to G, labeled "∄q" in gray.

See `examples/example-board-match.svg` for a stand-alone board illustration.

---

## 14. Coordinate System Note

All SVGs in the TicTacToeGameplay tutorial were created with Inkscape using the
PDF coordinate system where **y increases upward** (mathematical convention).
They are then stored with an Inkscape transform `matrix(1.3333333,0,0,-1.3333333, …)`
that flips the y-axis for screen display (SVG y increases downward). When
writing SVG directly (without this transform), use **screen coordinates** where
y increases downward. Internally, design the layout with y increasing downward;
translate your mental model accordingly.

---

## 15. TikZ Source Files and TikZ → SVG Translation

The commutative diagrams in the TicTacToeGameplay tutorial (naturality squares,
span diagrams, DPO grids, NAC triangles, pushout diagrams) were originally
typeset in LaTeX using `tikz-cd`. TikZ source files for each diagram are
included in this bundle under `tikz/`:

| File | Diagram |
|------|---------|
| `tikz/tikz-naturality.tex` | Naturality square for a morphism h : I → J |
| `tikz/tikz-span.tex` | Abstract span L ←l— K —r→ R |
| `tikz/tikz-dpo-diagram.tex` | Full DPO commutative grid (L K R / G D G′) |
| `tikz/tikz-nac-triangle.tex` | NAC triangle with dashed ∄q arrow |
| `tikz/tikz-pushout-universal.tex` | Pushout square with universal arrow to Q |
| `tikz/tikz-span-markx.tex` | mark_x span drawn with colored node circles (full TikZ) |
| `tikz/tikz-dpo-markx.tex` | mark_x DPO rewrite with colored board circles (full TikZ) |

### 15.1 Visual Correspondence: TikZ → SVG

When translating a tikz-cd diagram to SVG, the mapping is straightforward:

| tikz-cd construct | SVG equivalent |
|-------------------|---------------|
| Node label (e.g. `L`, `I(c)`) | `<text>` element at the node position |
| Horizontal/vertical arrow | `<line>` with `marker-end` filled-triangle arrowhead |
| Diagonal arrow | `<line>` at the appropriate angle, with `marker-end` |
| `\arrow[dashed]` | `<line stroke-dasharray="3 3">` in gray |
| Arrow label (e.g. `"f"`) | `<text>` near the arrow midpoint, slightly offset |
| `bend left` / `bend right` | `<path>` with a quadratic or cubic Bézier curve |
| Diagram column/row sep | Translate tikz column/row separations to SVG coordinate offsets |

For tikz-cd diagrams that show only morphism structure (no colored circles), the
SVG uses **text labels** at the corners with **directed arrows** between them and
no bounding boxes. For diagrams that include ACSets with node content (colored
circles), each tikz "node" becomes a **bounding box** containing the ACSet's
circle diagram as described in Section 4.

### 15.2 Compiling TikZ to SVG

**Method 1 (recommended): Inkscape CLI**
```bash
pdflatex tikz-span.tex
inkscape tikz-span.pdf --export-type=svg --export-plain-svg -o tikz-span.svg
```

**Method 2: pdf2svg**
```bash
pdflatex tikz-naturality.tex
pdf2svg tikz-naturality.pdf tikz-naturality.svg
```

**Method 3: dvisvgm**
```bash
latex tikz-naturality.tex
dvisvgm --pdf tikz-naturality.dvi -o tikz-naturality.svg
```

The Inkscape method (Method 1) produces the cleanest output, with paths rather
than text glyphs, matching the style of the tutorial SVGs. The resulting SVG
will carry a `matrix(...)` transform on the root group (Inkscape's coordinate
flip), which matches the tutorial SVGs exactly.

### 15.3 Adapting TikZ → Hand-Drawn SVG Style

The tutorial SVGs were styled beyond the raw TikZ output. When adapting a TikZ
diagram into a fully hand-drawn-style SVG:

1. Replace text-based corner labels with ACSet box content (circles for rows,
   per Section 5) — keep the arrow layout and labels from the TikZ diagram.
2. Use filled-triangle arrowheads in SVG (`<marker>` with a filled `<polygon>`).
3. Set arrow stroke to black at 0.8–1.0 pt.
4. Dashed arrows use `stroke-dasharray="3 3"` in gray.
5. Bounding boxes (Section 4) replace the implicit node boundaries of tikz-cd.

---

## 16. Quick Reference: Key Colors (Tutorial Palette)

The colors below are the tutorial's example palette. They are not prescriptive
requirements — see Section 2 for guidance on choosing alternative palettes.

| Element | Fill | Stroke |
|---------|------|--------|
| Regular (passive) Sq node | `#ccccff` | `#000099` |
| Interface / preserved Sq node (in image of K morphism) | `#7373ff` | `#0000cc` |
| New piece node (X or O, not in K) | `#ffbf80` | `#cc6600` |
| Bounding box background | `#f5f5f5` | `#a6a6a6` |
| Topology edge (RowE, ColE) | — | `#a6a6a6` |
| Diagonal edge (DiagE) | — | `#a6a6a6` (dashed) |
| Morphism arrow (between ACSets) | `#000000` | `#000000` |
| Dashed "does not exist" arrow | `#a6a6a6` | `#a6a6a6` |
