from typing import Any, List

from pydantic import parse_obj_as
from pydantic.error_wrappers import ValidationError

from web_server.ws_helpers.ws_models import UnclassifiedIssue

def parse_payload(payload: Any) -> List[UnclassifiedIssue]:
    try:
        issues: List[UnclassifiedIssue] = parse_obj_as(List[UnclassifiedIssue], payload)
        return issues
    except ValidationError as exc:
        raise exc