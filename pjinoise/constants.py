"""
constants
~~~~~~~~~

Common constants used within the pjinoise module. These tend to be
used to add clarity to the code, or at least that's the intention.
Whether it's successful is a question for the reader.
"""
X, Y, Z = 2, 1, 0
AXES = (X, Y, Z)
P = [151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
     140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247,
     120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57,
     177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74,
     165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60,
     211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65,
     25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200,
     196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
     52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
     207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
     119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
     129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
     218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
     241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106,
     157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205,
     93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156,
     180]
P.extend(P)

# Text for user interaction.
EN_TEXT = {
    # UI Status.
    'title': 'PJINOISE',
    'start': '{min:>4d}:{sec:>02d} Creating noise.',
    'end': '{min:>4d}:{sec:>02d} Noise created.',
    'noise': '{:>4d}:{:>02d} Creating noise generators.',
    'slice': '{:4d}:{:>02d} Starting slice {}.',
    'slice_end': '{:4d}:{:>02d} Completed slice {}.',
    'slices': '{:>4d}:{:>02d} Creating noise.',
    'slices_end': '{:>4d}:{:>02d} {} slices created.',
    'diff': '{:>4d}:{:>02d} Subtracting layer {}.',
    'filter': '{:>4d}:{:>02d} Running filter {}.',
    'filter_end': '{:>4d}:{:>02d} Ran filter {}.',
    'postprocess_start': '{:>4d}:{:>02d} Postprocessing noise.',
    'postprocess_end': '{:>4d}:{:>02d} Postprocessing complete.',
    'save_start': '{:>4d}:{:>02d} Saving {}.',
    'save_end': '{:>4d}:{:>02d} {} saved.',
    'gradient_error': 'Missing color value for gradient stop.',

    # Exception messages.
    'vol_dim_oob': 'Noise cannot have more generations than the table.',
    'table_or_size': ('Must provide either a table or an expected size '
                      'when initializing a {} object.'),
}
TEXT = EN_TEXT

# Valid data registration.
SUPPORTED_FORMATS = {
    'apng': 'PNG',
    'avi': 'AVI',
    'bmp': 'BMP',
    'gif': 'GIF',
    'jpeg': 'JPEG',
    'jpg': 'JPEG',
    'mp4': 'MP4',
    'png': 'PNG',
    'tif': 'TIFF',
    'tiff': 'TIFF',
    'webp': 'WebP',
}
VIDEO_FORMATS = {
    'AVI': 'MJPG',
    'MP4': 'mp4v',
}

# Color lookup.
COLOR = {
    # Don't colorize.
    '': [],

    # Grayscale
    'a': ['hsv(0, 0%, 100%)', 'hsv(0, 0%, 0%)'],
    'A': ['hsl(0, 0%, 75%)', 'hsl(0, 0%, 25%)'],

    # Electric blue.
    'b': ['hsv(200, 100%, 100%)', 'hsv(200, 100%, 0%)'],
    'B': ['hsl(200, 100%, 75%)', 'hsl(200, 100%, 25%)'],

    'bw': ['hsv(205, 100%, 100%)', 'hsv(200, 100%, 0%)'],
    'Bw': ['hsl(205, 100%, 75%)', 'hsl(200, 100%, 25%)'],

    'bk': ['hsv(200, 30%, 20%)', 'hsv(200, 30%, 0%)'],
    'BK': ['hsl(200, 30%, 30%)', 'hsl(200, 30%, 10%)'],

    # Cream
    'c': ['hsv(35, 100%, 100%)', 'hsv(35, 100%, 0%)'],
    'C': ['hsl(35, 100%, 80%)', 'hsl(35, 100%, 25%)'],

    'cw': ['hsv(30, 100%, 100%)', 'hsv(35, 100%, 0%)'],
    'Cw': ['hsl(30, 100%, 80%)', 'hsl(35, 100%, 25%)'],

    'cc': ['hsv(40, 100%, 100%)', 'hsv(35, 100%, 0%)'],
    'Cc': ['hsl(40, 100%, 80%)', 'hsl(35, 100%, 25%)'],

    'ck': ['hsv(35, 30%, 20%)', 'hsv(35, 30%, 0%)'],
    'CK': ['hsl(35, 30%, 30%)', 'hsl(35, 30%, 10%)'],

    # Dark.
    'k': ['hsv(220, 30%, 20%)', 'hsv(220, 30%, 0%)'],
    'K': ['hsl(220, 30%, 30%)', 'hsl(220, 30%, 10%)'],

    'kk': ['hsv(220, 30%, 10%)', 'hsv(220, 30%, 0%)'],
    'KK': ['hsl(220, 30%, 15%)', 'hsl(220, 30%, 5%)'],

    # Ectoplasmic teal.
    'e': ["hsv(190, 50%, 100%)", "hsv(190, 100%, 0%)"],
    'E': ["hsl(190, 50%, 100%)", "hsl(190, 100%, 30%)"],

    # Electric green.
    'g': ['hsv(90, 100%, 100%)', 'hsv(90, 100%, 0%)'],
    'G': ['hsl(90, 100%, 75%)', 'hsl(90, 100%, 25%)'],

    'gk': ['hsv(90, 30%, 20%)', 'hsv(90, 30%, 0%)'],
    'GK': ['hsl(90, 30%, 30%)', 'hsl(90, 30%, 10%)'],

    # Slate.
    'l': ['hsv(220, 30%, 50%)', 'hsv(220, 30%, 0%)'],
    'L': ['hsl(220, 30%, 75%)', 'hsl(220, 30%, 25%)'],

    # Electric pink.
    'p': ['hsv(320, 100%, 100%)', 'hsv(320, 100%, 0%)'],
    'P': ['hsl(320, 100%, 75%)', 'hsl(320, 100%, 25%)'],

    # Royal purple.
    'r': ['hsv(280, 100%, 100%)', 'hsv(280, 100%, 0%)'],
    'R': ['hsl(280, 100%, 75%)', 'hsl(280, 100%, 25%)'],

    'rw': ['hsv(285, 100%, 100%)', 'hsv(280, 100%, 0%)'],
    'Rw': ['hsl(285, 100%, 75%)', 'hsl(280, 100%, 25%)'],

    # Scarlet.
    's': ['hsv(350, 100%, 100%)', 'hsv(10, 100%, 0%)'],
    'S': ['hsl(350, 100%, 75%)', 'hsl(10, 100%, 25%)'],

    'sw': ['hsv(0, 100%, 100%)', 'hsv(10, 100%, 0%)'],
    'Sw': ['hsl(0, 100%, 75%)', 'hsl(10, 100%, 25%)'],

    'sk': ['hsv(350, 30%, 20%)', 'hsv(10, 30%, 0%)'],
    'SK': ['hsl(350, 30%, 30%)', 'hsl(10, 30%, 10%)'],

    # White.
    'w': ['hsv(0, 0%, 100%)', 'hsv(0, 0%, 0%)'],
    'W': ['hsl(0, 0%, 75%)', 'hsl(0, 0%, 25%)'],

    # Hue templates.
    't': ['hsv({}, 100%, 100%)', 'hsv({}, 100%, 0%)'],
    'T': ['hsl({}, 100%, 75%)', 'hsl({}, 100%, 25%)'],
}
