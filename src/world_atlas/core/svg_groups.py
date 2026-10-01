"""Replace owned SVG groups while preserving unrelated delivered coordinates."""

import re


def _group_ranges(markup,attribute,value):
    if attribute not in {"id", "data-tile-layer"}:
        raise ValueError("group ownership must use an SVG id or cartographic layer")
    opening = re.compile(r'<g\b[^>]*\b' + re.escape(attribute) + r'="' + re.escape(value) + r'"[^>]*>')
    ranges = []
    for found in opening.finditer(markup):
        depth = 0
        for token in re.finditer(r'</?g\b[^>]*>', markup[found.start():]):
            depth += -1 if token[0].startswith('</') else 1
            if depth == 0:
                ranges.append((found.start(), found.start() + token.end()))
                break
        else:
            raise ValueError(f"incomplete SVG group: {value}")
    return ranges


def extract_group(markup: str,attribute: str,value: str) -> str:
    ranges=_group_ranges(markup,attribute,value)
    if len(ranges)!=1:raise ValueError(f'expected one owned SVG group: {value}')
    start,end=ranges[0]
    return markup[start:end]


def replace_group(markup: str, attribute: str, value: str, replacement: str,
                  *, required: bool = False, before: str | None = None) -> str:
    ranges=_group_ranges(markup,attribute,value)
    if not ranges:
        if required:
            raise ValueError(f"missing owned SVG group: {value}")
        if not replacement:
            return markup
        anchor = (re.search(r'<g\b[^>]*\b' + re.escape(attribute) + r'="'
                            + re.escape(before) + r'"[^>]*>', markup) if before else None)
        position = anchor.start() if anchor else len(markup)
        return markup[:position] + replacement + markup[position:]
    result, previous = [], 0
    for index, (start, end) in enumerate(ranges):
        if start < previous:
            raise ValueError("owned groups must not nest inside each other")
        result.extend((markup[previous:start], replacement if index == 0 else ""))
        previous = end
    result.append(markup[previous:])
    return "".join(result)
