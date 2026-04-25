"""
Replace background of Architect images with solid Cocoon coral.
Strategy: extract subject (hair, skin, clothing, accessories) by their
distinctive characteristics — dark pixels OR skin tones (where G > B,
which lighter-coral isn't).
"""
import numpy as np
from PIL import Image, ImageFilter

TARGET_CORAL = (226, 80, 56)  # #E25038
W = H = 1024

def recolor(infile, outfile):
    orig = Image.open(infile).convert('RGB')
    if orig.size != (W, H):
        orig = orig.resize((W, H))
    arr = np.array(orig)
    R = arr[:, :, 0].astype(int)
    G = arr[:, :, 1].astype(int)
    B = arr[:, :, 2].astype(int)
    L = (np.maximum(R, np.maximum(G, B)) + np.minimum(R, np.minimum(G, B))) / 2.0 / 255.0

    # Subject mask:
    #   dark pixels (hair, clothing, dark eye details) — L < 0.45
    #   skin / gold / warm subject — G > B by 25+ (signal of yellow undertone)
    #     skin sample (187,139,73): G-B = 66 ✓
    #     gold              : G-B large ✓
    #     lighter-coral (243,105,91): G-B = 14 ✗ (excluded)
    #     cream  (246,247,238): G-B =  9 ✗ (excluded)
    is_dark = L < 0.45
    is_warm_subject = (G - B) > 25

    # Platinum-blonde hair detector. Sample colors:
    #   platinum (244,234,210): L=0.89, R-B=34, R-G=10  ← subject
    #   cream    (246,247,238): L=0.95, R-B= 8, R-G=-1  ← background (excluded by R-B<20)
    # Platinum hair detector — looser thresholds to catch more hair pixels.
    is_platinum = (L > 0.75) & ((R - B) >= 15) & ((R - B) < 70) & ((R - G) < 35)

    # Distance from center
    h, w = R.shape
    yy, xx = np.indices((h, w))
    dist = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)

    # PROTECTED CENTER (eyes, nose, lips): radius < 280, but NOT for
    # lighter-coral pixels (R clearly > G AND > B).
    is_lighter_coral_pixel = (R > G + 80) & (R > B + 80)
    protected_inner = (dist < 280) & ~is_lighter_coral_pixel

    subject_mask = is_dark | is_warm_subject | is_platinum | protected_inner

    # DILATE the subject mask by ~6 pixels to catch hair-edge pixels that
    # sit right next to platinum but whose color got shifted by the original
    # coral bleeding in (turns coral-tinted but is really hair).
    mask_pil = Image.fromarray(subject_mask.astype(np.uint8) * 255, mode='L')
    dilated = mask_pil.filter(ImageFilter.MaxFilter(size=13))
    subject_mask = np.array(dilated) > 128

    # Outer-ring cleanup — spare platinum hair which can extend into outer ring
    outer_ring = dist > 360
    outer_haze = outer_ring & (L > 0.55) & ~is_warm_subject & ~is_platinum
    subject_mask = subject_mask & ~outer_haze

    # Convert mask to PIL, slight smooth edge
    mask_img = Image.fromarray((subject_mask.astype(np.uint8) * 255), mode='L')
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1.2))

    # Solid coral background
    bg = Image.new('RGB', (W, H), TARGET_CORAL)

    # Composite
    result = Image.composite(orig, bg, mask_img)
    result.save(outfile, 'webp', quality=92)

    px = subject_mask.sum()
    total = subject_mask.size
    print(f"  {infile} -> {outfile}  (subject = {100*px/total:.1f}%)")

if __name__ == '__main__':
    for slug in ['zoe-1-architect', 'zoe-2-architect', 'zoe-3-architect']:
        recolor(f'{slug}.original.webp', f'{slug}.webp')
    print('done.')
