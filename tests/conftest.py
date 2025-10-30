import re

import pytest
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

    def open_port(self) -> None:
        self.is_open = True

    def close_port(self) -> None:
        self.is_open = False

    def reset_input_buffer() -> None:
        extended_serial.response_buffer.clear()

    base = ExtendedSerial.__bases__[0]
    mocker.patch.object(base, '__enter__', return_value=extended_serial)
    mocker.patch.object(base, 'write', side_effect=write)
    mocker.patch.object(base, 'read', side_effect=read)
    mocker.patch.object(base, 'open', new=open_port)
    mocker.patch.object(base, 'close', new=close_port)
    mocker.patch.object(base, 'reset_input_buffer', side_effect=reset_input_buffer)

    type(extended_serial).in_waiting = property(
        lambda self: len(extended_serial.response_buffer)
    )
    type(extended_serial).fd = None

    mocker.patch('bpod_core.com.ExtendedSerial', return_value=extended_serial)

    return extended_serial
