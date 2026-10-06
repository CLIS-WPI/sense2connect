# T6 second deployment: scene design (written and committed BEFORE any tracing)

Copy of results/TVT/T6/scene_design.md (results/ is never committed; this tracked copy is the
commit that precedes the traces). Fixed now; never changed after results are seen.

## Purpose
J5 asks whether the conclusions hold in a second deployment. The paper-1/2 street canyon has one
street, blockers that move along it, and reflectors that are all parallel to the street. A four-way
intersection is the canonical second urban geometry: cross traffic moves perpendicular to the main
street (blockers cross LoS paths at different angles and speeds), the O-RUs stand at corners,
building corners and cross-street facades add reflectors of the other orientation (x-facing), and
the UEs approach and pass a crossing. The reason for this choice is the geometry class, not any
result (no intersection result exists).

## Geometry (metres; x east, y north, z up; Mitsuba XML with box buildings, files under
configs/scenes/tvt_intersection/)
- Main street along x, facades at y = -8.6 (south) and y = +9.6 (north): the same 18.2 m width and
  lane/sidewalk layout as the paper-1 canyon.
- Cross street along y, facades at x = -9.1 (west) and x = +9.1 (east): 18.2 m wide.
- Four corner blocks (extend to |x|, |y| = 70 m): SW x in [-70, -9.1], y in [-70, -8.6], 25 m high,
  concrete; SE x in [9.1, 70], y in [-70, -8.6], 20 m, brick; NW x in [-70, -9.1], y in [9.6, 70],
  30 m, marble; NE x in [9.1, 70], y in [9.6, 70], 22 m, glass (the material set of the built-in
  simple_street_canyon, thickness 0.1 m).
- Ground: 200 m x 200 m concrete plane at z = 0.

## Radio units (same radios as paper 1: 28 GHz, 8x8 UPA lambda/2, isotropic elements, synthetic array,
paper orientation boresight +x, 1024 x 120 kHz subcarriers)
- O-RU 0: lamppost at the north-east corner (7.5, 7.0, 5.0) - mount name "corner".
- O-RU 1: lamppost at the south-west corner (-7.5, -7.0, 5.0) (paper-1 rule: second O-RU on the
  opposite sidewalk, x offset -15 m).
- Monostatic radar at O-RU 0 as in paper 1 (budget, CFAR, DBSCAN, map tracker unchanged; the
  image-method ghost rejection knows only the main-street walls y = 9.6 / -8.6, as in paper 1).

## Traffic (same densities, blocker classes, sizes and speeds as configs/m2_scenario.yaml; every
axis wraps on 80 m)
- Main street: eastbound lane y = -2.5, westbound y = +2.5 (x from -40 to 40).
- Cross street: northbound lane x = +2.5 (y from -40), southbound x = -2.5 (y from +40).
  Vehicles draw their lane uniformly from the four lanes (low: 4 cars, 1 bus, 1 truck; high: 12, 2, 2).
  Vehicles pass through the crossing without signal control (bodies are blockage screens, no
  collision model) - a stated simplification.
- Sidewalks: main south y = -7 (x from -40, +x), main north y = 6.5 (x from -40, +x), cross east
  x = +7 (from y = 6.5 southwards, -y), cross west x = -7 (from y = -7 northwards, +y); walkers and
  crossing pedestrians as paper 1 (low 6 + 2, high 12 + 4).
- UEs: ue-0 on the main south sidewalk (s0 = 20 m, 1.2 m/s, as paper 1), ue-1 on the cross east
  sidewalk (s0 = 20 m, 1.0 m/s), height 1.5 m.

## Seeds and runs
Mount "corner" (cache path results/cache/corner/<density>/seed_<s>, no collision with the canyon
caches), densities low and high; tuning seeds 101-105 (tuning only) and development seeds
1001-1010; 60 s, 0.1 s epochs, 10 ms comm steps; held-out seeds not used.

## What is evaluated there (T6)
The T4 tracker (pred_real vs none) against the paper-2 estimator and the PEB, and the T5 schemes
(A5, planner with the TVT tracker, risk-aware planner), with parameters tuned on the canyon tuning
seeds and, separately, re-tuned on the intersection tuning seeds; the learned predictor is applied as
trained on the canyon (generalisation).

## Known limitations (stated before tracing)
- Ghost rejection and the walkable map of paper 2 know the main street only; the TVT walkable area for
  the intersection is the union of the four sidewalk bands and the crossing (from this design).
- Arrays keep the paper orientation (+x).
