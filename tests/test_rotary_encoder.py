import ctypes
import logging
import struct

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
        re = RotaryEncoderModule('fake_port', encoder_resolution=256)
        assert re.hardware_version == mock_probe()
        assert re.clock_multiplier == 1 if mock_probe() == 1 else 4
        assert re.resolution == 256
        assert re.port == 'fake_port'
        mock_reset.assert_called_once()
        encoder_open.assert_called_once()

    def test_context_manager(self, mock_encoder, mocker):
        encoder_instance = mock_encoder('fake_port')
        encoder_close = mocker.spy(encoder_instance, 'close')
        with encoder_instance:
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

    def test_resolution(self, mock_encoder):
        assert mock_encoder('fake_port').resolution == 1024
        re = mock_encoder('fake_port', encoder_resolution=256)
        assert re.resolution == 256
        re.set_resolution(1024)
        assert re.resolution == 1024
        with pytest.raises(ValueError):
            re.set_resolution(-124)
        assert re._factor_deg_to_tic == re.resolution * re.clock_multiplier / 360
        assert re._factor_tic_to_deg == 1 / re._factor_deg_to_tic

    @pytest.mark.parametrize('degrees', [-135, 135], ids=lambda v: f'degrees={v}')
    def test_get_degrees(self, mock_encoder, mock_ext_serial, degrees):
        enc = mock_encoder('fake_port', encoder_resolution=1024)

        tics = enc._degrees_to_tics(degrees)
        degrees = enc._tics_to_degrees(tics)

        mock_ext_serial.mock_responses = {b'Q': ctypes.c_int16(tics)}
        assert enc.get_tics() == tics
        assert enc.get_degrees() == degrees

    @pytest.mark.parametrize('degrees', [-135, 135], ids=lambda v: f'degrees={v}')
    def test_set_degrees(self, mock_encoder, mock_ext_serial, caplog, degrees):
        enc = mock_encoder('fake_port', encoder_resolution=1024)

        tics1 = enc._degrees_to_tics(degrees)
        degrees1 = enc._tics_to_degrees(tics1)
        mock_ext_serial.mock_responses = {b'P' + ctypes.c_int16(tics1): b'\x01'}
        with caplog.at_level(logging.DEBUG):
            enc.set_degrees(degrees1)

        mock_ext_serial.mock_responses = {b'P' + ctypes.c_int16(tics1): b'\x00'}
        with pytest.raises(RuntimeError):
            enc.set_degrees(degrees1)

        tics2 = enc._degrees_to_tics(degrees * 100)
        degrees2 = enc._tics_to_degrees(tics2)
        with pytest.raises(ValueError):
            enc.set_degrees(degrees2)

    def test_wrap_point(self, mock_encoder, mock_ext_serial, caplog):
        enc = mock_encoder('fake_port', encoder_resolution=1024)
        enc._wrap_point_tics = enc._degrees_to_tics(180)
        assert enc.wrap_point == 180
        mock_ext_serial.mock_responses = {b'W' + ctypes.c_int16(128): b'\x01'}
        with caplog.at_level(logging.DEBUG):
            enc.set_wrap_point(-128 * enc._factor_tic_to_deg)
        assert mock_ext_serial.last_write == b'W' + ctypes.c_int16(128)
        assert len(caplog.records) == 1
        assert any('Setting wrap point' in r.message for r in caplog.records)
        mock_ext_serial.mock_responses = {struct.pack('<ch', b'W', 128): b'\x00'}
        with pytest.raises(RuntimeError):
            enc.set_wrap_point(128 * enc._factor_tic_to_deg)

    def test_zero(self, mock_encoder, mock_ext_serial, caplog):
        enc = mock_encoder('fake_port', encoder_resolution=1024)
        mock_ext_serial.mock_responses = {b'Z': b''}
        with caplog.at_level(logging.DEBUG):
            enc.zero()
        assert mock_ext_serial.last_write == b'Z'
        assert any('Resetting encoder position' in r.message for r in caplog.records)

    def test_reset_data_streams(self, mock_encoder, mock_ext_serial, caplog):
        enc = mock_encoder('fake_port', encoder_resolution=1024)
        enc._is_sd_logging = True
        mock_ext_serial.mock_responses = {b'X': b''}
        with caplog.at_level(logging.DEBUG):
            enc._reset_data_streams()
        assert mock_ext_serial.last_write == b'X'
        assert enc._is_sd_logging is False
        assert any('All data streams reset' in r.message for r in caplog.records)

    @pytest.mark.parametrize('hw_version', [1, 2], ids=lambda v: f'Rotary Encoder v{v}')
    def test_reset(self, hw_version, mocker):
        enc = RotaryEncoderModule.__new__(RotaryEncoderModule)
        enc._hardware_version = hw_version
        enc._degrees_to_tics = lambda x: int(x)
        enc._tics_to_degrees = lambda x: float(x)
        enc._serial = mocker.MagicMock()

        for name in [
            'set_event_transmission',
            'set_thresholds',
            'set_wrap_mode',
            'set_wrap_point',
            'set_stream_prefix',
        ]:
            mocker.patch.object(enc, name)

        enc.reset()
        enc.set_wrap_point.assert_called_once_with(180.0)
        enc.set_thresholds.assert_called_once_with([-40.0, 40.0])
        enc.set_wrap_mode.assert_called_once_with('bipolar')
        enc.set_event_transmission.assert_called_once_with(False)
        enc.set_stream_prefix.assert_not_called() if hw_version != 1 else None

    def test_thresholds(self, mock_encoder, mock_ext_serial, caplog):
        enc = mock_encoder('fake_port', encoder_resolution=1024)
        enc._wrap_point_tics = enc._degrees_to_tics(180)
        enc._thresholds = [-40.0, 40.0]
        assert enc._max_thresholds == 8
        assert enc.thresholds == [-40.0, 40.0]
        with pytest.raises(ValueError, match='cannot exceed .* wrap point'):
            enc.set_thresholds([-181.0, 40])
        with pytest.raises(ValueError, match=r'maximum of \d thresholds can be set'):
            enc.set_thresholds([x for x in range(enc._max_thresholds + 1)])
        thresholds_deg = [round(((x / 3.5) - 1) * 180) for x in range(8)]
        thresholds_tix = [enc._degrees_to_tics(x) for x in thresholds_deg]
        thresholds_array = (ctypes.c_int16 * 8)(*thresholds_tix)
        thresholds_bytes = ctypes.string_at(thresholds_array, 8)
        mock_ext_serial.mock_responses = {
            rb'T' + ctypes.c_uint8(8) + thresholds_bytes: b'\x01'
        }
        with caplog.at_level(logging.DEBUG):
            enc.set_thresholds(thresholds_deg)
        assert len(caplog.records) == 1
        assert any('Setting thresholds to' in r.message for r in caplog.records)
        assert enc.thresholds == [enc._tics_to_degrees(x) for x in thresholds_tix]
        mock_ext_serial.mock_responses = {
            rb'T'
            + ctypes.c_uint8(1)
            + ctypes.c_int16(enc._degrees_to_tics(42)): b'\x00'
        }
        with pytest.raises(RuntimeError, match='Failed to set thresholds'):
            enc.set_thresholds([42])
