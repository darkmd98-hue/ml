"""Small SVG helpers used by generated graph reports."""


def scale_points(values, width, height, padding, min_value, max_value):
    span = max_value - min_value or 1
    points = []
    for idx, value in enumerate(values):
        x = padding + (idx / max(len(values) - 1, 1)) * (width - 2 * padding)
        y = padding + (max_value - value) / span * (height - 2 * padding)
        points.append((round(x, 2), round(y, 2)))
    return points


def polyline(points):
    return ' '.join(f'{x},{y}' for x, y in points)


def make_svg_line(values, color, width, height, padding, min_value, max_value, dash=False):
    points = scale_points(values, width, height, padding, min_value, max_value)
    dash_attr = ' stroke-dasharray="8 6"' if dash else ''
    return (
        f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" '
        f'stroke-width="3"{dash_attr}/>'
    )
