# OpenDPD Studio logo prompt

**Current production artwork:** the raster draft below has been superseded by transparent SVG geometry and outlined Manrope lettering at the user’s request. See [vector artwork notes](opendpd-studio-vector-logo.md). The original image-generation prompts are retained as design history.

Brand references: [EMI Lab](https://www.tudemi.com/) and its [official logo](https://www.tudemi.com/images/emi-logo.svg), inspected 2026-09-12. Website CSS: ink `#101f31`, blue `#007dad`, cyan `#37b9e8`, paper `#ffffff`; Manrope sans serif.

Requested model: GPT Image 2.5. Execution: the built-in `image_gen` tool, which does not expose a model selector or a verifiable model version. Do not label the output as a verified GPT Image 2.5 generation.

## Exact generation prompt

```text
Use case: logo-brand.
Asset type: production logo for OpenDPD Studio, a scientific desktop and browser application from the Efficient Machine Intelligence Lab at TU Delft for neural power-amplifier modeling and digital predistortion.
Input images: Image 1 is the official EMI Lab logo, used only as a visual-family reference. Image 2 is the tudemi.com homepage, used only for its typography, palette and clean academic design. Ignore all photographs and all page copy. Create a new original product logo; do not copy the EMI wordmark or the TU Delft flame.
Primary request: design ONE finished horizontal OpenDPD Studio logo, with a compact emblem on the left and a precise two-line wordmark on the right. The emblem should combine a strong open circular O silhouette with a simple continuous radio waveform and three circuit-trace terminals. The circuitry should feel related to EMI's brain-and-circuit emblem, but simpler and legible at 24 pixels. Convey intelligent signal linearization through the waveform becoming a clean, controlled line. Use balanced geometry, generous negative space and a few confident rounded strokes.
Text, verbatim: "OpenDPD" as the large first line, spelled O-p-e-n-D-P-D with exactly that capitalization. "STUDIO" as the much smaller second line in gently tracked uppercase, left-aligned with OpenDPD. No other words, letters or slogans. Use a bold, contemporary geometric sans serif in the family of Manrope and the EMI lettering, with carefully balanced kerning. All lettering must be accurately readable.
Color palette: deep ink navy #101f31 for the wordmark and structural strokes, Delft blue #007dad as the emblem's principal color, and a restrained cyan #37b9e8 waveform accent. Use only flat solid colors.
Composition: a tightly framed horizontal lockup, approximately 3:1 overall aspect ratio, emblem height equal to the combined two-line wordmark height, vertically centered, with a clear gap between emblem and wordmark and small even outer padding. A transparent canvas with real alpha, no background rectangle, no checkerboard drawn into the image. Aim for 1536 by 512 pixels and crisp vector-like edges.
Style: refined Dutch academic technology identity, precise, minimal, calm and modern. A functional app identity, not an illustration.
Avoid: gradients, glows, shadows, 3D, bevels, metallic textures, mockup scenes, extra badges, ornamental frames, generic AI sparkles, robots, antenna towers, Wi-Fi symbols, tiny dense circuit details, all photographs, extra variations, a logo presentation sheet, and watermarks. Return only the one logo on transparency.
```

## Exact corrective edit prompt

The first output drew a checkerboard instead of alpha and was rejected. The selected asset uses a white background suitable for a white brand tile in the navigation and the About page.

```text
Use case: background-extraction / logo-brand production cleanup. Edit the supplied OpenDPD Studio logo. Preserve the exact geometry, emblem, letter shapes, spelling OpenDPD and STUDIO, alignment, proportions and flat navy/blue/cyan colors. Change only the backdrop and finish: replace the ENTIRE gray checkerboard, including every hole inside letters and every open gap in the symbol, with perfectly uniform solid pure white #FFFFFF. Remove all fabric wrinkles, paper texture, lighting, shadows, highlights and gradients. Flatten the emblem and text into crisp perfectly uniform solid colors: #101f31 navy, #007dad blue, #37b9e8 cyan. The result must be a clean flat digital logo asset on a pure white canvas, never a photograph or mockup. Keep the same tightly framed 3:1 horizontal composition, with small even outer margins. Do not draw a checkerboard. Do not change or add any text. Return one finished white-background logo only.
```

Selected original: `exec-3031235d-f7ef-44ca-bc4a-3d5c2a220066.png`, 2172 × 724 pixels. Former project asset: `frontend/src/assets/opendpd-studio-logo.png` (replaced by SVG). The original generated image remains in the image-generation output directory.
