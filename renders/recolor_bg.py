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

    # Catch platinum-blonde hair: very light, near center (subject area).
    # Platinum  (244,234,210): L=0.89, near center → mask
    # Cream — typically lives in image corners (radius > 480), not here.
    h, w = R.shape
    yy, xx = np.indices((h, w))
    dist = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    is_inner_light = (dist < 380) & (L > 0.78)

    subject_mask = is_dark | is_warm_subject | is_inner_light

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
