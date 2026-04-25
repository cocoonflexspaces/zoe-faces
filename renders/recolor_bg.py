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
    # Distance from center (needed for spatial constraints)
    h, w = R.shape
    yy, xx = np.indices((h, w))
    dist = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)

    # Platinum hair detector — spatially constrained to subject area.
    # Hair doesn't live in image corners (dist > 500); confining the detector
    # there eliminates speckles from JPEG-compressed cream misclassification.
    is_platinum = (
        (L > 0.75)
        & ((R - B) >= 15)
        & ((R - B) < 70)
        & ((R - G) < 35)
        & (dist < 450)
    )

    # PROTECTED CENTER (eyes, nose, lips): radius < 280, but NOT for
    # lighter-coral pixels (R clearly > G AND > B).
    is_lighter_coral_pixel = (R > G + 80) & (R > B + 80)
    protected_inner = (dist < 280) & ~is_lighter_coral_pixel

    # Clean the platinum mask:
    # 1. Erode (MinFilter) to kill isolated pixels (JPEG-compressed cream)
    # 2. Dilate (MaxFilter) to catch hair-edge pixels next to real platinum
    # Result: only connected platinum regions expand, stray pixels disappear.
    platinum_pil = Image.fromarray(is_platinum.astype(np.uint8) * 255, mode='L')
    eroded = platinum_pil.filter(ImageFilter.MinFilter(size=5))
    platinum_dilated = np.array(eroded.filter(ImageFilter.MaxFilter(size=19))) > 128

    subject_mask = is_dark | is_warm_subject | platinum_dilated | protected_inner

    # Outer-ring cleanup — spare real platinum hair (dilated) that extends out
    outer_ring = dist > 360
    outer_haze = outer_ring & (L > 0.55) & ~is_warm_subject & ~platinum_dilated
    subject_mask = subject_mask & ~outer_haze

    # Convert mask to PIL.
    mask_img = Image.fromarray((subject_mask.astype(np.uint8) * 255), mode='L')
    # Median filter kills isolated speckles.
    mask_img = mask_img.filter(ImageFilter.MedianFilter(size=5))
    # ERODE the mask ~3 px: the rim pixels of the subject (shoulders/hair
    # edges) carry lighter-coral bleed from ERNIE's rendering. Shrinking
    # the mask replaces that bleed ring with solid coral — no shadow line.
    mask_img = mask_img.filter(ImageFilter.MinFilter(size=7))
    # Feather transition.
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2.5))

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
