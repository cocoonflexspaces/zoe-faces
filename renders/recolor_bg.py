"""
Replace cream + lighter-coral background pixels in Architect Zoe images
with the exact Cocoon coral #E25038. Subject pixels (dark hair, brown skin,
black clothing, gold accessories) are spared.
"""
import numpy as np
from PIL import Image
import sys

TARGET_CORAL = np.array([226, 80, 56], dtype=np.uint8)  # #E25038

def recolor(infile, outfile):
    img = np.array(Image.open(infile).convert('RGB'))
    R = img[:, :, 0].astype(int)
    G = img[:, :, 1].astype(int)
    B = img[:, :, 2].astype(int)

    # Lightness (HSL midpoint)
    max_c = np.maximum(np.maximum(R, G), B)
    min_c = np.minimum(np.minimum(R, G), B)
    L = (max_c + min_c) / 2.0 / 255.0

    # Saturation
    delta = max_c - min_c

    # SPATIAL classification: combine pixel color with distance from center.
    # The subject sits roughly within radius 380 of the 1024×1024 center.
    # Outside that radius we can aggressively replace anything bg-like.
    h, w = R.shape
    yy, xx = np.indices((h, w))
    cy, cx = h / 2, w / 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # CREAM: must be near-neutral (R≈G, both close to B). This excludes
    # platinum-blonde hair which is light but R is clearly higher than G.
    #   cream ≈ (246, 247, 238): R-G ≈ -1, near neutral
    #   platinum hair ≈ (244, 234, 210): R-G ≈ 10  ← spared
    is_cream = (L > 0.82) & (np.abs(R - G) <= 6) & (np.abs(R - B) <= 30)

    # LIGHTER CORAL (the soft halo around the subject): R clearly dominant.
    #   light coral ≈ (243, 105, 91): R-G = 138, R-B = 152
    #   skin        ≈ (187, 139, 73): R-G =  48, R-B = 114  ← spared
    is_lighter_coral = (L > 0.50) & (R > G + 80) & (R > B + 80)

    # OUTER region (radius > 360 from center): broaden lighter-coral
    # detection because no subject pixels are out there.
    outer = dist > 360
    outer_bg = outer & ((L > 0.50) & (R > G + 30) & (R > B + 30))

    # Outer-region boundary cleanup: catch the cream→coral transition haze
    # without touching platinum hair (which is close-to-neutral light).
    #   boundary haze ~ R clearly > G AND > B, e.g., (235, 200, 180) — R-G=35
    #   platinum hair  ~ (244, 234, 210) — R-G=10  ← spared
    outer_boundary = outer & (L > 0.55) & (R > G + 30) & (R > B + 50)

    is_bg = is_cream | is_lighter_coral | outer_bg | outer_boundary

    out = img.copy()
    out[is_bg] = TARGET_CORAL

    Image.fromarray(out).save(outfile, 'webp', quality=92)
    print(f"  {infile} -> {outfile}  (changed {is_bg.sum()} px / {img.shape[0]*img.shape[1]} px)")

if __name__ == '__main__':
    for slug in ['zoe-1-architect', 'zoe-2-architect', 'zoe-3-architect']:
        recolor(f'{slug}.original.webp', f'{slug}.webp')
    print("done.")
