# MicroDuck Studio · Design QA

Date: 2026-10-27. Selected direction: **reference 03**. This is the new Studio application, independent of the previous visualization UI.

## Comparison evidence

- Source visual truth: [reference-03.png](apps/microduck-studio/design/reference-03.png), 1487 × 1058 pixels, no browser chrome.
- Browser-rendered implementation: [home.png](docs/microduck/studio-images/home.png), 1488 × 1056 CSS/pixel viewport, deviceScaleFactor 1, light theme, skill-library route, signed-in local workspace. Captured from `http://10.156.16.169:8771/` with Chrome DevTools Protocol.
- The source and final browser screenshot were opened **together in the same image comparison input**. Their dimensions differ by 1 px horizontally and 2 px vertically; neither is a double-density capture. Composition was compared at equivalent scale without treating this negligible frame difference as a layout defect.
- Additional rendered states: [scene editor](docs/microduck/studio-images/scene-editor.png), [actual MuJoCo preview](docs/microduck/studio-images/preview.png), [mobile](docs/microduck/studio-images/mobile.png). Mobile viewport: 390 × 844 CSS/pixels, deviceScaleFactor 1.
- The source depicts an empty workspace; the implementation has saved skills and therefore adds “我的技能” before workspace activity. These are real entries, not source-matching fabricated activity.
- Full-view and mobile captures resolve the sidebar labels, card captions, buttons and body copy at native density. No additional cropped region was needed for those checks.

## Findings and comparison history

1. **[P2, fixed] Initial desktop scale and image treatment.** The [first browser render](docs/microduck/studio-images/home-before.png) used a smaller display heading/card rhythm than the selected mock and an awkward image crop. The display scale, spacing and media regions were adjusted. The final full-view comparison preserves the narrow rail, large heading, two main cards and primary-action hierarchy.
2. **[P2, fixed] Mobile action wrapping.** The card footer squeezed the action buttons into vertical text. The mobile footer now wraps its groups, keeps button text on one line and retains usable target sizes. The final 390 px capture has no horizontal overflow or overlapping controls.
3. **[P2, fixed] Missing/empty resource handling.** Browser checks found a favicon 404 and a training-metrics request returning an error before any metrics existed. A Phosphor-based favicon and an empty-metrics response resolve both. The final public-address interaction check reports no console errors.
4. **[P3, accepted source limitation] Historical recording sharpness.** The historical card uses a frame from the real evidence video. It is visibly less sharp than the generated mock. The preview preserves that actual robot/scene and links to the real recording; a generated high-resolution robot is not substituted as execution evidence. Future recordings can improve resolution.

## Required fidelity surfaces

| Surface | Inspection and disposition |
|---|---|
| Fonts / typography | A Chinese system sans-serif stack, bold display heading, medium-weight card titles and muted body copy retain the source hierarchy. The raster mock has no font specification, so exact family matching is not asserted. Desktop line breaks and mobile heading/button wrapping were inspected after correction. |
| Spacing / layout rhythm | Narrow persistent rail, generous main margins, two wide cards, restrained corner radii and a vertical workflow preserve the selected composition. Saved skills add genuine content below the hero. Mobile becomes one column and keeps controls reachable. |
| Colors / tokens | Cool pale background, deep teal primary controls, white surfaces and muted blue-gray supporting text follow reference 03. Success/experimental states use semantic labels in addition to color. No decorative gradient or fake activity is added. |
| Images / icons | Real historical video frame on the first card; generated ball/box/toy illustration on the template card. The latter is a template illustration, not evidence of learned capability. Phosphor icons supply the navigation and brand mark; no handcrafted SVG replacements or CSS-drawn product imagery. |
| Copy / content | Chinese product labels describe tasks and user actions. “历史开发版本”, experimental statuses, preview/replay labels and worker availability reflect actual evidence. Advanced model/log details live inside workflow pages. “在真实的仿真任务中” avoids implying hardware validation. |

## Interactions and accessibility checked

- Open skill library and play actual historical video (`video.readyState >= 1`).
- Create a skill, save a task, drag an object, undo the move, submit a preview, and display the returned MuJoCo render.
- Run a real simulation through the queue, receive at least three distinct MJPEG frames, cancel it and verify the cancelled state.
- Review desktop and mobile layouts; verify `scrollWidth <= innerWidth` at 390 px.
- Use semantic button/input labels, visible focus styles and modal Escape/focus handling. This is a focused usability check, not a full assistive-technology audit.
- Chrome runtime/log console checks: no errors in the final pass. TypeScript and production Vite build pass separately.

## Open questions and remaining scope

No unresolved P0/P1/P2 visual finding. The mock has no detailed mobile/editor/error-state specification; those screens extend its layout and tokens. Training-model quality, GPU worker availability, full deployment reproducibility and skill acceptance are tracked separately in [studio-validation.md](docs/microduck/studio-validation.md), and are not implied by visual QA.

## Implementation checklist

- [x] Use selected direction 03 in a separate React application and service.
- [x] Fix desktop typography, card spacing and media treatment.
- [x] Fix mobile wrapping and verify no horizontal overflow.
- [x] Keep real recordings and actual simulation previews distinct.
- [x] Re-capture and compare source and implementation together after fixes.
- [x] Verify key browser interactions and console output.
- [ ] Optional P3 polish: re-record historical demos at higher resolution when those models are next evaluated.

final result: passed
