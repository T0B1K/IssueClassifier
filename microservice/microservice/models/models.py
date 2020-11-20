"""Pydantic models for issues.

This class contains two models: IndexedIssue to denote indexed issues with
their regular string representation of the issue body, and VectorisedIssue
which is largely the same as IndexedIssue except for the fact that the issue
body consists of a numpy array. This numpy array represents the feature
vectors of the corresponding IndexedIssue body with the same index.
"""
from typing import Any, List, Union

import ujson
from pydantic import BaseModel


class IndexedIssue(BaseModel):
    """Class for indexed issues.

    An indexed issue consists of an index, an issue body, and the list of
    labels.

    The index can either be an int or string, and pydantic will attempt
    to cast the input to an int. If this fails, it resorts to casting the index
    into a string for compatibility.

    The issue body should be a JSON string, which will be parsed into a native
    Python string.

    Labels do not need to be provided, but can be filled if desired. With that
    being said, this has no effect on the vectoriser and classifiers or their output.

    Args:
        BaseModel (BaseModel): The pydantic base model.
    """

    index: Union[int, str]
    body: str
    labels: List[str] = []

    class Config:
        """Configuration of the IndexedIssue class.

        The current configuration sets the json_loads function to the
        ujson.loads function from ujson, which is faster when it comes to
        de-/serialisation of data.
        """

        json_loads = ujson.loads


class VectorisedIssue(BaseModel):
    """Class for transformed issues.

    Transformed issues are very similar to IndexedIssue instances. The
    difference lies in the data type of the body, which is a numpy array
    representing the feature vectors produced by the vectoriser.

    The index can either be an int or string, and pydantic will attempt
    to cast the input to an int. If this fails, it resorts to casting the index
    into a string for compatibility.

    Args:
        BaseModel (BaseModel): The pydantic base model.
    """

    index: Union[int, str]
    body: Any
    labels: List[str] = []

    class Config:
        """Configuration of the VectorisedIssue class.

        The current configuration sets the json_loads function to the
        ujson.loads function from ujson, which is faster when it comes to
        de-/serialisation of data.

        In addition, since numpy.ndarray does not provide validators
        out-of-the-box, pydantic is set to accept arbitrary data types to
        circumvent this issue.
        """

        json_loads = ujson.loads
        arbitrary_types_allowed = True
