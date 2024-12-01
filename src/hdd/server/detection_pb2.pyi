from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SUCCESS: _ClassVar[Status]
    FAILURE: _ClassVar[Status]
SUCCESS: Status
FAILURE: Status

class DetectionRequest(_message.Message):
    __slots__ = ("width", "height", "image", "mask")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    image: bytes
    mask: bytes
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., image: _Optional[bytes] = ..., mask: _Optional[bytes] = ...) -> None: ...

class DescriptionRequest(_message.Message):
    __slots__ = ("coords", "width", "height", "image", "mask")
    COORDS_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    coords: _containers.RepeatedCompositeFieldContainer[Coordinates]
    width: int
    height: int
    image: bytes
    mask: bytes
    def __init__(self, coords: _Optional[_Iterable[_Union[Coordinates, _Mapping]]] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., image: _Optional[bytes] = ..., mask: _Optional[bytes] = ...) -> None: ...

class DetectionResponse(_message.Message):
    __slots__ = ("status", "coordinates_list")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    COORDINATES_LIST_FIELD_NUMBER: _ClassVar[int]
    status: ResponseStatus
    coordinates_list: _containers.RepeatedCompositeFieldContainer[Coordinates]
    def __init__(self, status: _Optional[_Union[ResponseStatus, _Mapping]] = ..., coordinates_list: _Optional[_Iterable[_Union[Coordinates, _Mapping]]] = ...) -> None: ...

class DescriptionResponse(_message.Message):
    __slots__ = ("status", "confidence_list")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_LIST_FIELD_NUMBER: _ClassVar[int]
    status: ResponseStatus
    confidence_list: _containers.RepeatedCompositeFieldContainer[Confidence]
    def __init__(self, status: _Optional[_Union[ResponseStatus, _Mapping]] = ..., confidence_list: _Optional[_Iterable[_Union[Confidence, _Mapping]]] = ...) -> None: ...

class Coordinates(_message.Message):
    __slots__ = ("x1", "y1", "x2", "y2")
    X1_FIELD_NUMBER: _ClassVar[int]
    Y1_FIELD_NUMBER: _ClassVar[int]
    X2_FIELD_NUMBER: _ClassVar[int]
    Y2_FIELD_NUMBER: _ClassVar[int]
    x1: float
    y1: float
    x2: float
    y2: float
    def __init__(self, x1: _Optional[float] = ..., y1: _Optional[float] = ..., x2: _Optional[float] = ..., y2: _Optional[float] = ...) -> None: ...

class Confidence(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[ConfidenceEntry]
    def __init__(self, entries: _Optional[_Iterable[_Union[ConfidenceEntry, _Mapping]]] = ...) -> None: ...

class ConfidenceEntry(_message.Message):
    __slots__ = ("name", "confidence")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    name: str
    confidence: float
    def __init__(self, name: _Optional[str] = ..., confidence: _Optional[float] = ...) -> None: ...

class ResponseStatus(_message.Message):
    __slots__ = ("success", "err_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: Status
    err_message: str
    def __init__(self, success: _Optional[_Union[Status, str]] = ..., err_message: _Optional[str] = ...) -> None: ...
