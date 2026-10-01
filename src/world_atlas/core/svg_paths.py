"""Compact SVG coordinates with bounded browser-relative accumulation."""

COORDINATE_SCALE = 100_000_000
_MAX_RELATIVE_VERTICES = 64


def _coordinate(value: int) -> str:
    if not value:
        return "0"
    sign = "-" if value < 0 else ""
    whole, fraction = divmod(abs(value), COORDINATE_SCALE)
    return sign + (str(whole) if whole else "") + (
        "." + f"{fraction:08d}".rstrip("0") if fraction else "")


def _separator(previous: str, following: str) -> str:
    # SVG consumes each number greedily. A sign always starts another number;
    # a second decimal point does too, but `3` followed by `.5` needs a space.
    return "" if following[0] == "-" or (
        following[0] == "." and "." in previous) else " "


def integer_subpath_data(points, *, closed: bool) -> str:
    """Encode delivered integer vertices without changing any coordinate.

    SVG viewers may accumulate relative coordinates at single precision.
    Absolute anchors protect each bounding edge (including map/tile frames)
    and bound every remaining relative chain to 64 vertices. Neighbouring
    features therefore retain their exact delivered boundary endpoints.
    """
    points = list(points)
    if closed and len(points) > 1 and points[-1] == points[0]:
        points.pop()
    if len(points) < (3 if closed else 2):
        return ""
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    previous = points[0]
    sx, sy = _coordinate(previous[0]), _coordinate(previous[1])
    result = ["M", sx, _separator(sx, sy), sy]
    last_coordinate = sy
    command = None
    relative_vertices = 0
    for x, y in points[1:]:
        absolute = x in (min_x, max_x) or y in (min_y, max_y) or relative_vertices >= _MAX_RELATIVE_VERTICES
        next_command = "L" if absolute else "l"
        dx, dy = (x, y) if absolute else (x - previous[0], y - previous[1])
        sx, sy = _coordinate(dx), _coordinate(dy)
        result.append(next_command if next_command != command else
                      _separator(last_coordinate, sx))
        result.extend((sx, _separator(sx, sy), sy))
        last_coordinate = sy
        previous = (x, y)
        command = next_command
        relative_vertices = 0 if absolute else relative_vertices + 1
    if closed:
        result.append("Z")
    return "".join(result)
