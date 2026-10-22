"""Lightweight JSON report writer.

Mirrors the surface of Neon's ``neon.tool.report.Report`` (``add_member`` /
``add_member_vector`` / ``write``) so benchmark and example scripts written
against Neon run unchanged on Carbon. Neon backed this by a C++ rapidjson
writer; Carbon has no such need for the report path, so this is a pure-Python
implementation that serialises to ``<name>.json``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


class Report:
    def __init__(
        self,
        report_name: str = "report",
    ) -> None:
        self.report_name = report_name
        self._members: Dict[str, Any] = {}

    def add_member(
        self,
        name: str,
        value: Any,
    ) -> None:
        if not isinstance(value, (int, float, str)):
            raise TypeError(f"Report.add_member expects int/float/str, got {type(value)!r}")
        self._members[name] = value

    def add_member_vector(
        self,
        name: str,
        value_vec: List[Any],
    ) -> None:
        if not isinstance(value_vec, list):
            raise TypeError("Report.add_member_vector expects a list")
        if value_vec:
            first = type(value_vec[0])
            if not all(isinstance(v, first) for v in value_vec):
                raise ValueError("All items in the vector must share one type")
            if first not in (int, float):
                raise TypeError(f"Unsupported vector element type: {first!r}")
        self._members[name] = list(value_vec)

    def write(
        self,
        name: str,
        also_print: bool = True,
    ) -> None:
        payload = {self.report_name: self._members}
        path = name if name.endswith(".json") else f"{name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        if also_print:
            print(json.dumps(payload, indent=2))
