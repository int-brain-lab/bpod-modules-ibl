import logging

import pytest
from serial import Serial, SerialException

from bpod_modules_ibl.rotary_encoder_module import RotaryEncoderModule


@pytest.fixture(params=[1, 2], ids=lambda v: f'Rotary Encoder v{v}')
def mock_probe(mocker, request):
    return mocker.patch.object(RotaryEncoderModule, 'probe', return_value=request.param)


@pytest.fixture
def mock_reset(mocker):
    return mocker.patch.object(RotaryEncoderModule, 'reset')


@pytest.fixture
def mock_encoder(mock_probe, mock_reset, mock_ext_serial):
    return RotaryEncoderModule


class TestRotaryEncoder:
    @pytest.mark.parametrize('hw_version', [1, 2], ids=lambda v: f'Rotary Encoder v{v}')
    def test_probe_valid(self, mock_ext_serial, hw_version):
        serial_response = b'\xd9\x01' if hw_version == 1 else b'\xd9\x00'
        mock_ext_serial.mock_responses = {b'CI\xfa': serial_response}
        assert RotaryEncoderModule.probe('fake_port') == hw_version

    def test_probe_timeout(self, mock_ext_serial):
        mock_ext_serial.mock_responses = {b'.*': b''}
        assert RotaryEncoderModule.probe('fake_port', raise_value_error=False) is None
        with pytest.raises(ValueError) as e:
            RotaryEncoderModule.probe('fake_port')
        assert isinstance(e.value.__cause__, TimeoutError)

    def test_probe_invalid(self, mock_ext_serial):
        mock_ext_serial.mock_responses = {b'.*': b'invalid'}
        assert RotaryEncoderModule.probe('fake_port', raise_value_error=False) is None
        with pytest.raises(ValueError) as e:
            RotaryEncoderModule.probe('fake_port')
        assert isinstance(e.value.__cause__, NotImplementedError)

    def test_serial_exception(self, mock_ext_serial):
        mock_ext_serial.__enter__.side_effect = SerialException('could not open port')
        with pytest.raises(SerialException, match='Is the device connected'):
            RotaryEncoderModule.probe('invalid_port', raise_value_error=True)
        with pytest.raises(SerialException, match='Is the device connected') as e:
            RotaryEncoderModule.probe('invalid_port', raise_value_error=False)
        assert isinstance(e.value.__cause__, SerialException)

        mock_ext_serial.__enter__.side_effect = SerialException('another exception')
        with pytest.raises(SerialException) as e:
            RotaryEncoderModule.probe('fake_port')
        assert e.value.__cause__ is None

    def test_init(self, mock_probe, mock_reset, mock_ext_serial, mocker):
        encoder_open = mocker.spy(RotaryEncoderModule, 'open')
        encoder = RotaryEncoderModule('fake_port')
        assert encoder.hardware_version == mock_probe()
        assert encoder.clock_multiplier == {1: 1, 2: 4}[mock_probe()]
        assert encoder.port == 'fake_port'
        mock_reset.assert_called_once()
        encoder_open.assert_called_once()

    def test_context_manager(self, mock_encoder, mocker):
        encoder_close = mocker.spy(mock_encoder, 'close')
        with mock_encoder('fake_port'):
            encoder_close.assert_not_called()
        encoder_close.assert_called_once()

    def test_open_close(self, mock_encoder, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        encoder = mock_encoder('fake_port')
        serial_open = mocker.spy(Serial, 'open')
        serial_close = mocker.spy(Serial, 'close')

        caplog.clear()
        for _ in range(2):
            encoder.close()
            assert serial_close.call_count == 1
        assert 'Closing serial connection' in caplog.text
        assert len(caplog.records) == 1

        caplog.clear()
        for _ in range(2):
            encoder.open()
            assert serial_open.call_count == 1
        assert 'Opening serial connection' in caplog.text
        assert len(caplog.records) == 1
