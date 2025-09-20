from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class EditAction:
    kind: str
    target_func: str = "main"
    details: Dict = field(default_factory=dict)

@dataclass
class EditPlan:
    actions: List[EditAction]
    summary: str

@dataclass(frozen=True)
class ImportFromRename:
    module: str
    old: str
    new: str

@dataclass(frozen=True)
class ModuleAttributeRename:
    module: str
    old_attr: str
    new_attr: str

@dataclass(frozen=True)
class FunctionDocstringSpec:
    function: str
    doc: str
