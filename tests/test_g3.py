"""Unit tests for the G3 (PALAS/DOMUS, PROSOFT 5) additions - no central required.

The expected values come from the examples in "TDS15132 DoIP API" (2026):
sensor reports are one big-endian IEEE-754 float, temperatures are Kelvin.

Run with:
    pytest tests/test_g3.py -v
"""
from __future__ import annotations

import asyncio
import struct
import sys
from pathlib import Path

import pytest

# Import client.py directly - avoids homeassistant imports from __init__.py.
sys.path.insert(0, str(Path(__file__).parent.parent / "custom_components" / "teletask"))

from client import (  # noqa: E402
    CMD_SUBSCRIBE,
    CMD_SUBSCRIBE_GROUP,
    G3_UNSUPPORTED_FUNCTIONS,
    SUBSCRIBE_GROUP_MAX,
    STATE_ON,
    TELETASK_STX,
    FunctionCode,
    TeletaskClient,
    decode_float,
    decode_sensor,
)


def _client() -> TeletaskClient:
    client = TeletaskClient("localhost", 55957, lambda _: None)
    client._central_id = 1
    return client


# ---------------------------------------------------------------------------
# Floats and sensors (examples from the DoIP API)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        (bytes([0x45, 0x26, 0x40, 0x00]), 2660.0),
        (bytes([0x42, 0x54, 0x00, 0x00]), 53.0),
        (bytes([0x45, 0xA2, 0xA0, 0x00]), 5204.0),
        (bytes([0x40, 0x20, 0x00, 0x00]), 2.5),
        (bytes([0x43, 0x95, 0xC0, 0x00]), 299.5),
    ],
)
def test_decode_float_matches_api_examples(raw, expected):
    assert decode_float(raw) == expected


def test_decode_float_ignores_trailing_bytes():
    # Reports carry more bytes after the value; they "can be ignored".
    assert decode_float(bytes([0x45, 0x26, 0x40, 0x00, 0x00, 0x00, 0x0E])) == 2660.0


def test_decode_float_is_not_an_integer():
    assert decode_float(bytes([0x45, 0x26, 0x40, 0x00])) != 1160134656


@pytest.mark.parametrize("raw", [b"", b"\x45", b"\x45\x26\x40"])
def test_decode_float_needs_four_bytes(raw):
    assert decode_float(raw) is None


@pytest.mark.parametrize(
    "raw",
    [struct.pack(">f", float("nan")), struct.pack(">f", float("inf"))],
)
def test_decode_float_rejects_nan_and_infinity(raw):
    assert decode_float(raw) is None


def test_g3_temperature_subtracts_273():
    result = decode_sensor("TEMPERATURE", bytes([0x43, 0x95, 0xC0, 0x00]), g3=True)
    assert result == {"state": 26.5, "value": 26.5}


def test_g3_light_humidity_power_pass_value_through():
    assert decode_sensor("LIGHT", bytes([0x45, 0x26, 0x40, 0x00]), g3=True)["state"] == 2660.0
    assert decode_sensor("HUMIDITY", bytes([0x42, 0x54, 0x00, 0x00]), g3=True)["state"] == 53.0
    assert decode_sensor("GAS", bytes([0x40, 0xB2, 0x03, 0x21]), g3=True)["state"] == 5.56


def test_g3_small_light_value():
    assert decode_sensor("LIGHT", bytes([0x40, 0xB6, 0x66, 0x66]), g3=True)["state"] == 5.7


def test_g3_bad_value_gives_no_state_instead_of_a_wrong_one():
    nan = struct.pack(">f", float("nan"))
    assert decode_sensor("TEMPERATURE", nan, g3=True) == {"state": None, "value": None}
    assert decode_sensor("TEMPERATURE", b"\x43\x95", g3=True) == {"state": None, "value": None}


def test_g3_climate_zones_are_not_decoded():
    # The API does not say whether the 273 offset applies to climate floats.
    assert decode_sensor("TEMPERATURECONTROL", bytes(20), g3=True) is None


def test_g2_decoding_is_unchanged_by_default():
    # 2-byte layout: humidity 53 -> 0x0035
    assert decode_sensor("HUMIDITY", bytes([0x00, 0x35])) == {"state": 53, "value": 53}


def test_g3_pulse_counter_is_whole_number():
    assert decode_sensor("PULSECOUNTER", struct.pack(">f", 1234.0), g3=True)["state"] == 1234


# ---------------------------------------------------------------------------
# A full report frame, as the central sends it
# ---------------------------------------------------------------------------

def _sensor_report(number: int, value_bytes: bytes) -> bytes:
    # [STX, LEN, 0x10, CID, FN, NUM_HI, NUM_LO, ERROR, STATE..., CS]
    body = bytes([TELETASK_STX, 0, 0x10, 1, int(FunctionCode.SENSOR),
                  (number >> 8) & 0xFF, number & 0xFF, 0]) + value_bytes + bytes(4)
    body = bytes([body[0], len(body)]) + body[2:]
    return body + bytes([sum(body) & 0xFF])


def test_sensor_report_frame_decodes_as_g3_temperature():
    client = _client()
    frame = _sensor_report(1, bytes([0x43, 0x95, 0xC0, 0x00]))
    event, used = client._parse_frame(bytearray(frame))
    assert used == len(frame)
    assert event is not None
    assert event.function == FunctionCode.SENSOR and event.number == 1
    assert decode_sensor("TEMPERATURE", event.raw, g3=True)["state"] == 26.5


# ---------------------------------------------------------------------------
# Subscriptions (0x22 / 0x23)
# ---------------------------------------------------------------------------

def _capture(client: TeletaskClient) -> list[bytes]:
    sent: list[bytes] = []

    async def fake_send(data: bytes) -> None:
        sent.append(data)

    client._send = fake_send  # type: ignore[method-assign]
    return sent


def _check_frame(frame: bytes) -> None:
    assert frame[0] == TELETASK_STX
    assert frame[1] == len(frame) - 1          # LENGTH counts STX + LENGTH + payload, not the checksum
    assert frame[-1] == sum(frame[:-1]) & 0xFF


def test_single_device_uses_command_0x22():
    client = _client()
    sent = _capture(client)
    asyncio.run(client.subscribe_devices(int(FunctionCode.RELAY), [7]))
    assert len(sent) == 1
    frame = sent[0]
    _check_frame(frame)
    assert frame[2] == CMD_SUBSCRIBE
    assert frame[3:8] == bytes([1, int(FunctionCode.RELAY), 0, 7, STATE_ON])


def test_several_devices_use_command_0x23():
    client = _client()
    sent = _capture(client)
    asyncio.run(client.subscribe_devices(int(FunctionCode.RELAY), [1, 7, 300]))
    assert len(sent) == 1
    frame = sent[0]
    _check_frame(frame)
    assert frame[2] == CMD_SUBSCRIBE_GROUP
    assert frame[3:-1] == bytes([1, int(FunctionCode.RELAY), 0, 1, 0, 7, 1, 44, STATE_ON])


def test_large_groups_are_split_to_fit_the_length_byte():
    client = _client()
    sent = _capture(client)
    numbers = list(range(1, 2 * SUBSCRIBE_GROUP_MAX + 2))      # 201 devices
    asyncio.run(client.subscribe_devices(int(FunctionCode.RELAY), numbers))
    assert len(sent) == 3
    for frame in sent:
        _check_frame(frame)
        assert len(frame) <= 255
    # The last chunk holds a single device, sent as 0x22.
    assert sent[-1][2] == CMD_SUBSCRIBE


def test_unsupported_functions_on_g3():
    assert G3_UNSUPPORTED_FUNCTIONS == {FunctionCode.INPUT, FunctionCode.DISPLAYMESSAGE}


# ---------------------------------------------------------------------------
# Frames without checksum (G3 centrals)
# ---------------------------------------------------------------------------

def _client_no_checksum() -> TeletaskClient:
    client = TeletaskClient("localhost", 55957, lambda _: None, send_checksum=False)
    client._central_id = 1
    return client


def test_g3_keep_alive_is_exactly_the_documented_three_bytes():
    assert _client_no_checksum()._build_keep_alive() == bytes([0x02, 0x03, 0x0B])


def test_g3_state_request_has_no_trailing_checksum():
    frame = _client_no_checksum()._frame(bytes([0x06, 0x01, 0x02, 0x00, 0x52]))
    assert frame == bytes([0x02, 0x07, 0x06, 0x01, 0x02, 0x00, 0x52])
    assert frame[1] == len(frame)


def test_default_client_still_appends_checksum():
    frame = _client()._build_keep_alive()
    assert frame == bytes([0x02, 0x03, 0x0B, 0x10])


def test_unknown_reply_from_palas_is_consumed_without_event():
    # Observed on a real Palas (serial 721002404): reply 0x0E to any request.
    client = _client()
    buf = bytearray([0x02, 0x04, 0x0E, 0x00, 0x14])
    event, used = client._parse_frame(buf)
    assert event is None
    assert used == 5
