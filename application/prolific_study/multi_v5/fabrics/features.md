# Fabrics — Feature Table

**Tested concept:** Bracklin
**Foil concept:** Calosse
**Distractor concepts:** Spurnek, Thalwick (retired — no longer used; see notes)

Feature roles: **T** = tested (in questions), **P** = padding (control/repeat only), **D** = distractor concept feature

| Feature | Role | Bracklin | Calosse | Spurnek | Thalwick |
|---------|------|----------|---------|---------|---------|
| Raw material (animal/plant/etc.) | T | Fleece of the Grennock mountain sheep | Stalk fibers of the cassoweed plant | Extruded petroleum-based filament | Processed bark pulp of the thalwort shrub |
| Fiber class (animal-based / plant-based / synthetic) | T | Animal-based | Plant-based | Synthetic | Plant-based |
| Weaving or construction technique | T | Double-weft interlock weaving | Plain-weave tabby | Warp-knit construction | Diagonal twill binding |
| Pile or surface structure | unused | Short-napped with a brushed raised pile | Smooth, flat surface with no pile | Uniform micro-loop surface | Ribbed corded surface |
| Primary uses | T | Winter outerwear and ceremonial robes | Summer garments and decorative wall hangings | Athletic wear and moisture-wicking underlayers | Upholstery and formal drapery |
| Natural color range | T | Warm ivory to ash grey | Pale sage green to straw yellow | Translucent white (no natural pigment) | Pale buff to dark ochre |
| Dye behavior | T | Absorbs deep, even color with acid dyes; resists discharge printing | Takes reactive dyes readily; prone to gradual sunlight fading | Accepts disperse dyes only; poor wash fastness | Holds fiber-reactive dyes moderately; darkens with prolonged bath time |
| Texture | P | Dense and slightly felted | Lightweight and slightly crisp | Smooth and slippery | Firm and slightly rough |
| Geographic origin | P | Highland uplands of the Veldrath range | Lowland river valleys of the Cassine basin | Industrial production facilities | Scrubland plateaus of the interior |
| Thermal property | P | Highly insulating; traps warmth without added bulk | Breathable and cooling in warm weather | Moisture-wicking; temperature-neutral | Moderately insulating; retains body heat |
| Loom type | D | — | — | Rapier loom | Jacquard loom |
| Care instructions | D | — | — | Machine washable; low-heat tumble dry | Dry clean only; avoid prolonged moisture exposure |
| Trade history | D | — | — | Developed for industrial sportswear markets in the mid-twentieth century | Historically traded along coastal merchant routes in exchange for spices and pigments |
| Weight per m² | D | — | — | 180–220 g/m² | 320–400 g/m² |

## Notes

- **Bracklin** is the tested concept: the six T-role features above appear in the
  study texts (control/repeat_short/repeat_long/distractor) and are the subject of
  questions.yaml. It was chosen over Calosse because it already had a working sentence-level
  draft and its T-role features (fiber class, weaving mechanism, uses, color, dye behavior)
  produce clean 2×2 designs (e.g. fiber class × primary use for Q4) without relying on any
  fact that is a bare prohibition.
- **QFB05** was changed from a none-of-the-above trick question to a genuine 2-correct
  multi-select without introducing any new fact into control. It now pairs Bracklin with two
  facts already present in control but not previously tested standalone: natural color range
  (only used before as part of the dye-behavior question) and primary uses (only used before
  inside QFB04's 2×2 combo).
- **Pile or surface structure** was dropped from control (and all other conditions, including
  the distractor filler paragraph) during a later text edit and is no longer tested anywhere.
  QFB02 was repointed from pile/color to the two sub-claims of the weaving-mechanism sentence
  (interlocking + no stitching) so every question stays answerable from control alone. The
  pile row is kept here only as a historical record.
- **Calosse** provides wrong-answer foils for Bracklin's T-role features in questions.yaml
  (e.g. "plain-weave tabby" as a construction-technique foil, "pale sage green to straw
  yellow" as a color foil). Calosse does not appear in any study text.
- **Spurnek** and **Thalwick** were the distractor-condition paragraph-2 concepts in the
  previous version of this topic. That design violated the README rule against introducing a
  second named concept in the distractor's filler paragraph, so they have been dropped from
  the texts. Their D-role rows are kept here only as a historical record; they are not used
  anywhere in the current texts or questions.
- The distractor condition's second paragraph instead uses untested Bracklin-only padding
  facts (texture, geographic origin, thermal property — the P-role rows above) plus a few
  invented production-process details (finishing time, hand-fulling, storage/rolling) that
  are not testable and do not name a second concept.
- Arbitrary terms that must follow the 1×/2×/3× repetition pattern (control/repeat_short/
  repeat_long) because they are recall targets in questions.yaml: "double-weft interlock
  weaving" and "discharge printing."
