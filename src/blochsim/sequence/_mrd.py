"""Reading the sequence description an MRD stream carries ahead of its data.

Pulserver's reconstruction proxy infers the description from the sequence file
a series was played from and sends it after the MRD XML header and before the
first acquisition, as ISMRMRD waveforms that every MRD writer keeps with the
series: per file of the sequence chain a header (``waveform_id`` 1024) and an
event list (1025), and one waveform per RF definition (1026), each one vector
of 32-bit words led by its length. What is read here is mechanical; which
signal model the rows drive is chosen by whoever reads them.
"""

from __future__ import annotations

__all__ = ["description_from_waveforms", "read_mrd_description"]

import struct
from collections.abc import Iterable
from typing import Any, BinaryIO

import numpy as np

from ._description import (
    AdcRole,
    EventType,
    RfDefinition,
    RfShape,
    RfUse,
    SequenceDescription,
    SequenceEvent,
    _trapezoid,
)

# MRD streaming message identifiers, and the fixed size of a FILENAME message.
_FILENAME, _CONFIG, _HEADER, _CLOSE, _TEXT = 1, 2, 3, 4, 5
_WAVEFORM = 1026
_FILENAME_BYTES = 1024
# An ISMRMRD waveform header: its size, and where it holds the sample count,
# the channel count and the waveform id.
_WAVEFORM_HEADER_BYTES = 40
_SAMPLES_AT, _CHANNELS_AT, _ID_AT = 28, 30, 36

#: Waveform ids of a description: header, event list, RF definition.
HEADER_ID, EVENTS_ID, RF_DEFINITION_ID = 1024, 1025, 1026
#: Words an event takes in the event list.
_EVENT_WORDS = 10


def read_mrd_description(
    stream: BinaryIO,
) -> tuple[str | None, list[SequenceDescription]]:
    """Read an MRD stream up to its sequence description.

    The messages ahead of the description -- a config, a config file name, the
    XML header, texts, other waveforms such as the physiological traces -- are
    read and the stream is left at the message after the description's last
    waveform, which is where its first acquisition is.

    Parameters
    ----------
    stream : binary file-like
        The stream, read with ``read(n)``: an open file, or a socket's
        ``makefile("rb")``.

    Returns
    -------
    header : str or None
        The MRD XML header, ``None`` if the stream carried none before the
        description.
    descriptions : list of SequenceDescription
        One per file of the sequence chain, in play order. RF definitions carry
        the envelope normalised to a peak of one, played at the event's
        ``amplitude_hz``, and their sample times in units of the RF raster.

    Raises
    ------
    ValueError
        If any other message, or the end of the stream, comes before the
        description.
    """
    header = None
    collected: list[tuple[int, np.ndarray]] = []
    while True:
        (identifier,) = struct.unpack("<H", _exactly(stream, 2))
        if identifier == _FILENAME:
            _exactly(stream, _FILENAME_BYTES)
        elif identifier in (_CONFIG, _HEADER, _TEXT):
            (length,) = struct.unpack("<I", _exactly(stream, 4))
            text = _exactly(stream, length).split(b"\0", 1)[0].decode("utf-8")
            if identifier == _HEADER:
                header = text
        elif identifier == _WAVEFORM:
            head = _exactly(stream, _WAVEFORM_HEADER_BYTES)
            (samples,) = struct.unpack_from("<H", head, _SAMPLES_AT)
            (channels,) = struct.unpack_from("<H", head, _CHANNELS_AT)
            (waveform_id,) = struct.unpack_from("<H", head, _ID_AT)
            data = np.frombuffer(_exactly(stream, 4 * samples * channels), "<u4")
            if waveform_id in (HEADER_ID, EVENTS_ID, RF_DEFINITION_ID):
                collected.append((waveform_id, data))
                if _complete(collected):
                    return header, description_from_waveforms(collected)
        else:
            raise ValueError(
                f"MRD message {identifier} arrived before the sequence description"
            )


def description_from_waveforms(waveforms: Iterable[Any]) -> list[SequenceDescription]:
    """Read the descriptions a series' waveforms carry.

    Parameters
    ----------
    waveforms : iterable
        The series' waveforms: ``ismrmrd.Waveform`` objects, as an HDF5 file
        read with ``ismrmrd.Dataset.read_waveform`` gives them, or
        ``(waveform_id, data)`` pairs. Those of other ids, such as the
        physiological traces, are passed over.

    Returns
    -------
    list of SequenceDescription
        One per file of the sequence chain, in play order; empty when the
        waveforms carry none.

    Raises
    ------
    ValueError
        If a description's header names more RF definitions or events than
        its waveforms carry.
    """
    headers: dict[int, np.ndarray] = {}
    events: dict[int, np.ndarray] = {}
    definitions: dict[int, list[np.ndarray]] = {}
    for item in waveforms:
        identifier, data = (
            item if isinstance(item, tuple) else (item.waveform_id, item.data)
        )
        words = _words(data)
        if int(identifier) == HEADER_ID:
            headers[int(words[1])] = words
        elif int(identifier) == EVENTS_ID:
            events[int(words[0])] = words
        elif int(identifier) == RF_DEFINITION_ID:
            definitions.setdefault(int(words[0]), []).append(words)
    return [
        _from_words(headers[index], events.get(index), definitions.get(index, []))
        for index in sorted(headers)
    ]


def _words(data: Any) -> np.ndarray:
    """The word vector of a description waveform, its leading length dropped."""
    flat = np.ascontiguousarray(np.asarray(data), dtype="<u4").reshape(-1)
    return flat[1 : 1 + int(flat[0])]


def _complete(collected: list[tuple[int, np.ndarray]]) -> bool:
    """Whether every subsequence's header, events and RF definitions have arrived."""
    headers = [_words(data) for kind, data in collected if kind == HEADER_ID]
    if not headers or len(headers) < int(headers[0][2]):
        return False
    events = sum(kind == EVENTS_ID for kind, _ in collected)
    definitions = sum(kind == RF_DEFINITION_ID for kind, _ in collected)
    return events == len(headers) and definitions == sum(int(h[4]) for h in headers)


def _from_words(
    header: np.ndarray, events: np.ndarray | None, definitions: list[np.ndarray]
) -> SequenceDescription:
    index = int(header[1])
    if events is None or len(definitions) != int(header[4]):
        raise ValueError(
            f"subsequence {index}: the header names {int(header[4])} RF definitions "
            f"and its waveforms carry {len(definitions)}, with "
            f"{'no' if events is None else 'an'} event list"
        )
    tr_duration_us, raster_s = (float(v) for v in header[5:9].view("<f8"))
    rows = events[2:].reshape(int(events[1]), _EVENT_WORDS)
    kinds = rows[:, 0].astype(np.int32)
    times_us = rows[:, 1:3].copy().view("<f8").reshape(-1)
    params = rows[:, 3:].copy().view("<f4")
    return SequenceDescription(
        subsequence_index=index,
        tr_duration_us=tr_duration_us,
        events=tuple(
            _event(EventType(int(kind)), float(time_us), row)
            for kind, time_us, row in zip(kinds, times_us, params, strict=True)
        ),
        rf_definitions={
            int(words[1]): _definition_from_words(words, raster_s)
            for words in definitions
        },
        rf_raster_time_s=raster_s,
    )


def _definition_from_words(words: np.ndarray, raster_s: float) -> RfDefinition:
    floats = words.view("<f4")
    at, counts = 14, []
    for _ in range(3):
        given, uncompressed, stored = (int(v) for v in words[at : at + 3])
        counts.append((uncompressed, stored) if given else None)
        at += 3
    shapes = []
    for count in counts:
        if count is None:
            shapes.append(None)
            continue
        shapes.append(RfShape(count[0], floats[at : at + count[1]].copy()))
        at += count[1]
    magnitude, phase, time = shapes
    return _built_definition(
        identifier=int(words[1]),
        bandwidth_hz=float(floats[2]),
        num_bands=int(words[3]),
        band_offsets=tuple(float(v) for v in floats[4:12]),
        band_bandwidth_hz=float(floats[12]),
        magnitude=magnitude,
        phase=phase,
        time=time,
        raster_s=raster_s,
    )


def _exactly(stream: BinaryIO, count: int) -> bytes:
    read = stream.read(count)
    if len(read) != count:
        raise ValueError("the MRD stream ended before the sequence description")
    return read


def _event(kind: EventType, time_us: float, row: np.ndarray) -> SequenceEvent:
    if kind is EventType.RF:
        return SequenceEvent.rf(
            time_us,
            definition_id=int(row[0]),
            use=RfUse(int(row[1])),
            amplitude_hz=float(row[2]),
            phase_rad=float(row[3]),
            frequency_hz=float(row[4]),
            shim_id=int(row[5]),
            slice_select_gradient_hz_per_m=float(row[6]),
        )
    if kind is EventType.ADC:
        return SequenceEvent.adc(
            time_us,
            role=AdcRole(int(row[0])),
            phase_rad=float(row[1]),
            is_echo=bool(row[2]),
        )
    return SequenceEvent.wait(time_us)


def _built_definition(
    *,
    identifier: int,
    bandwidth_hz: float,
    num_bands: int,
    band_offsets: tuple[float, ...],
    band_bandwidth_hz: float,
    magnitude: RfShape,
    phase: RfShape | None,
    time: RfShape | None,
    raster_s: float,
) -> RfDefinition:
    envelope = magnitude.decompress().astype(np.float64)
    time_s = (
        (np.arange(envelope.size) + 0.5) * raster_s
        if time is None
        else time.decompress(scale=raster_s).astype(np.float64)
    )
    return RfDefinition(
        id=identifier,
        bandwidth_hz=bandwidth_hz,
        num_bands=num_bands,
        band_frequency_offsets_hz=band_offsets,
        band_bandwidth_hz=band_bandwidth_hz,
        # Not carried by the stream: the stored envelope's squared integral.
        total_b1sq_power=float(_trapezoid(envelope**2, time_s)),
        magnitude=magnitude,
        phase=phase,
        time=time,
    )
