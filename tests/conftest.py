import re
from unittest.mock import PropertyMock

import pytest
import serial
from bpod_core.com import ExtendedSerial
from serial import Serial


@pytest.fixture
def mock_serial(mocker):
    return mocker.MagicMock(spec=Serial)


@pytest.fixture
def mock_ext_serial(mocker):
    """Mock base class methods for ExtendedSerial."""
    extended_serial = ExtendedSerial()
    extended_serial.response_buffer = bytearray()
    extended_serial.mock_responses = {}
    extended_serial.last_write = b''

    def write(data) -> None:
        for pattern, value in extended_serial.mock_responses.items():
            if re.match(pattern, data):
                extended_serial.response_buffer.extend(value)
                extended_serial.last_write = data
                return
        raise AssertionError(f'No matching response for input {data}')

    def read(size: int = 1) -> bytes:
        response = bytes(extended_serial.response_buffer[:size])
        del extended_serial.response_buffer[:size]
        return response

    def in_waiting() -> int:
        return len(extended_serial.response_buffer)

    def open_port(self) -> None:
        self.is_open = True

    def close_port(self) -> None:
        self.is_open = False

    mocker.patch.object(serial.Serial, '__enter__', return_value=extended_serial)
    mocker.patch.object(serial.Serial, 'write', side_effect=write)
    mocker.patch.object(serial.Serial, 'read', side_effect=read)
    mocker.patch.object(serial.Serial, 'open', new=open_port)
    mocker.patch.object(serial.Serial, 'close', new=close_port)
    mocker.patch.object(serial.Serial, 'reset_input_buffer')
    mocker.patch.object(
        serial.Serial, 'in_waiting', new_callable=PropertyMock, side_effect=in_waiting
    )
    return extended_serial
