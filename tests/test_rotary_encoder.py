import ctypes
import logging
import struct

import numpy as np
import pytest
from serial import Serial, SerialException

from bpod_modules_ibl import RotaryEncoderModule
from bpod_modules_ibl._rotary_encoder import DTYPE_LOGGING_OUT, MAX_N_THRESHOLDS


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
        """Probe returns the correct hardware version for a valid device response."""
        serial_response = b'\xd9\x01' if hw_version == 1 else b'\xd9\x00'
        mock_ext_serial.mock_responses = {b'CI\xfa': serial_response}
        assert RotaryEncoderModule.probe('fake_port') == hw_version

    def test_probe_timeout(self, mock_ext_serial):
        """Probe returns None or raises ValueError when the device does not respond."""
        mock_ext_serial.mock_responses = {b'.*': b''}
        assert RotaryEncoderModule.probe('fake_port', raise_value_error=False) is None
        with pytest.raises(ValueError) as e:
            RotaryEncoderModule.probe('fake_port')
        assert isinstance(e.value.__cause__, TimeoutError)

    def test_probe_invalid(self, mock_ext_serial):
        """Probe returns None or raises ValueError for an unrecognised response."""
        mock_ext_serial.mock_responses = {b'.*': b'invalid'}
        assert RotaryEncoderModule.probe('fake_port', raise_value_error=False) is None
        with pytest.raises(ValueError) as e:
            RotaryEncoderModule.probe('fake_port')
        assert isinstance(e.value.__cause__, NotImplementedError)

    def test_serial_exception(self, mock_ext_serial):
        """Probe re-raises SerialException without wrapping it in a ValueError."""
        mock_ext_serial.__enter__.side_effect = SerialException('could not open port')
        with pytest.raises(SerialException, match='could not open port'):
            RotaryEncoderModule.probe('invalid_port', raise_value_error=True)

        mock_ext_serial.__enter__.side_effect = SerialException('another exception')
        with pytest.raises(SerialException) as e:
            RotaryEncoderModule.probe('fake_port')
        assert e.value.__cause__ is None

    def test_init(self, mock_probe, mock_reset, mock_ext_serial, mocker):
        """Constructor sets hardware attributes and opens the serial connection."""
        encoder_open = mocker.spy(RotaryEncoderModule, 'open')
        re = RotaryEncoderModule('fake_port', encoder_resolution=256)
        assert re.hardware_version == mock_probe()
        assert re.clock_multiplier == 1 if mock_probe() == 1 else 4
        assert re.resolution == 256
        assert re.port == 'fake_port'
        mock_reset.assert_called_once()
        encoder_open.assert_called_once()

    def test_open_close(self, mock_encoder, mocker, caplog):
        """Opening and closing are idempotent and log once per state change."""
        caplog.set_level(logging.DEBUG)
        encoder = mock_encoder('fake_port')
        serial_open = mocker.spy(Serial, 'open')
        serial_close = mocker.spy(Serial, 'close')

        caplog.clear()
        for _ in range(2):
            encoder.close()
            assert serial_close.call_count == 1
        assert 'Closing connection to' in caplog.text
        assert len(caplog.records) == 1

        caplog.clear()
        for _ in range(2):
            encoder.open()
            assert serial_open.call_count == 1
        assert 'Opening connection to ' in caplog.text
        assert len(caplog.records) == 1

    def test_resolution(self, mock_encoder):
        """Resolution and conversion factors update correctly when set."""
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
        """get_degrees returns the current encoder position in degrees."""
        enc = mock_encoder('fake_port', encoder_resolution=1024)

        tics = enc._degrees_to_tics(degrees)
        degrees = enc._tics_to_degrees(tics)

        mock_ext_serial.mock_responses = {b'Q': ctypes.c_int16(tics)}
        assert enc.get_tics() == tics
        assert enc.get_degrees() == degrees

    @pytest.mark.parametrize('degrees', [-135, 135], ids=lambda v: f'degrees={v}')
    def test_set_degrees(self, mock_encoder, mock_ext_serial, caplog, degrees):
        """set_degrees moves the encoder to the given position in degrees."""
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
        """set_wrap_point sends the correct command and logs the new value."""
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
        """zero resets the encoder position and logs the action."""
        enc = mock_encoder('fake_port', encoder_resolution=1024)
        mock_ext_serial.mock_responses = {b'Z': b''}
        with caplog.at_level(logging.DEBUG):
            enc.zero()
        assert mock_ext_serial.last_write == b'Z'
        assert any('Resetting encoder position' in r.message for r in caplog.records)

    def test_reset_data_streams(self, mock_encoder, mock_ext_serial, caplog):
        """_reset_data_streams sends the stop command and clears SD logging flag."""
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
        """reset initialises encoder to default state with correct calls."""
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
        """set_thresholds validates inputs and sends values to hardware."""
        enc = mock_encoder('fake_port', encoder_resolution=1024)
        enc._wrap_point_tics = enc._degrees_to_tics(180)
        enc._thresholds = [-40.0, 40.0]
        assert MAX_N_THRESHOLDS == 8
        assert enc.thresholds == [-40.0, 40.0]
        with pytest.raises(ValueError, match=r'cannot exceed .* wrap point'):
            enc.set_thresholds([-181.0, 40])
        with pytest.raises(ValueError, match=r'maximum of \d thresholds can be set'):
            enc.set_thresholds(list(range(MAX_N_THRESHOLDS + 1)))
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

    def test_enable_thresholds(self, mock_encoder, mock_ext_serial, caplog):
        """enable_thresholds accepts bool, string, and list inputs."""
        enc = mock_encoder('fake_port', encoder_resolution=1024)
        enc._wrap_point_tics = enc._degrees_to_tics(180)
        enc._thresholds = [-40.0, 40.0]
        mock_ext_serial.mock_responses = {rb'.*': b''}
        caplog.set_level(logging.DEBUG)

        enc.enable_thresholds(True)
        assert mock_ext_serial.last_write == struct.pack('<cB', b';', 255)
        assert 'Enabled all' in caplog.text
        caplog.clear()
        enc.enable_thresholds(False)
        assert mock_ext_serial.last_write == struct.pack('<cB', b';', 0)
        assert 'Disabled all' in caplog.text

        caplog.clear()
        enc.enable_thresholds('00001000')
        assert mock_ext_serial.last_write == struct.pack('<cB', b';', 8)
        assert 'Enabled threshold 3' in caplog.text
        with pytest.raises(ValueError):
            enc.enable_thresholds('12345678')
        with pytest.raises(ValueError):
            enc.enable_thresholds('asd')

        caplog.clear()
        enc.enable_thresholds([0, 2])
        assert mock_ext_serial.last_write == struct.pack('<cB', b';', 5)
        assert 'Enabled thresholds 0, 2' in caplog.text
        with pytest.raises(ValueError):
            enc.enable_thresholds([9])

        with pytest.raises(TypeError):
            enc.enable_thresholds({'key': 'value'})

    def test_properties(self, mock_encoder):
        """Read-only properties reflect the correct internal state."""
        enc = mock_encoder('fake_port')
        enc._event_transmission = True
        assert enc.event_transmission is True
        assert enc.sd_logging is False
        assert enc.sd_logging == enc._is_sd_logging
        assert enc.wrap_mode == enc._wrap_mode
        assert enc.is_usb_streaming is False

    def test_exit(self, mock_encoder, mocker):
        """Exiting the context manager disables SD logging."""
        with mock_encoder('fake_port') as enc:
            spy = mocker.spy(RotaryEncoderModule, 'set_sd_logging')
        spy.assert_any_call(enc, False)

    def test_process_stream(self, mock_encoder, caplog):
        """_process_stream dispatches position and event data to callbacks."""
        enc = mock_encoder('fake_port', encoder_resolution=1024)

        # Position — no callback (early return)
        tics, us = 512, 1_000_000
        data = struct.pack('<chI', b'P', tics, us)
        enc._process_stream(data)

        # Position — with callback
        position_calls = []
        enc.set_position_callback(lambda t, d: position_calls.append((t, d)))
        enc._process_stream(data)
        assert len(position_calls) == 1
        assert position_calls[0][0] == us
        assert position_calls[0][1] == pytest.approx(tics * enc._factor_tic_to_deg)

        # Event — no callback (early return)
        event_type, event_code, evt_us = 1, 2, 500_000
        data = struct.pack('<c2BI', b'E', event_type, event_code, evt_us)
        enc._process_stream(data)

        # Event — with callback
        event_calls = []
        enc.set_event_callback(lambda et, ec, t: event_calls.append((et, ec, t)))
        enc._process_stream(data)
        assert event_calls == [(event_type, event_code, evt_us)]

        # Unknown message type
        caplog.clear()
        data = struct.pack('<c2BI', b'X', event_type, event_code, evt_us)
        with caplog.at_level(logging.ERROR):
            enc._process_stream(data)
        assert any('Unknown message type' in r.message for r in caplog.records)

    def test_set_sd_logging(self, mock_encoder, mock_ext_serial, caplog):
        """set_sd_logging enables/disables logging with hardware version guard."""
        enc = mock_encoder('fake_port')

        # No-op when state is already False
        assert enc._is_sd_logging is False
        enc.set_sd_logging(False)
        mock_ext_serial.write.assert_not_called()

        # v2: RuntimeError when trying to enable
        if enc.hardware_version == 2:
            with pytest.raises(RuntimeError, match='not supported'):
                enc.set_sd_logging(True)
            return

        # v1: enable logging
        mock_ext_serial.mock_responses = {b'L': b''}
        with caplog.at_level(logging.DEBUG):
            enc.set_sd_logging(True)
        assert enc._is_sd_logging is True
        assert 'SD Logging enabled' in caplog.text

        # No-op when already True
        mock_ext_serial.write.reset_mock()
        enc.set_sd_logging(True)
        mock_ext_serial.write.assert_not_called()

        # Disable logging
        caplog.clear()
        mock_ext_serial.mock_responses = {b'F': b''}
        with caplog.at_level(logging.DEBUG):
            enc.set_sd_logging(False)
        assert enc._is_sd_logging is False
        assert 'SD Logging disabled' in caplog.text

    def test_get_logged_data(self, mocker):
        """get_logged_data returns structured data with uint32 rollover correction."""
        raw_dtype = np.dtype([('tics', np.int32), ('time', np.uint32)])

        # v2 should raise
        enc = RotaryEncoderModule.__new__(RotaryEncoderModule)
        enc._hardware_version = 2
        enc._serial_device_name = 'Rotary Encoder Module'
        with pytest.raises(RuntimeError, match='not supported'):
            enc.get_logged_data()

        def make_v1_enc():
            e = RotaryEncoderModule.__new__(RotaryEncoderModule)
            e._hardware_version = 1
            e._is_sd_logging = False
            e._serial_device_name = 'Rotary Encoder Module'
            e._factor_tic_to_deg = 360.0 / (1024 * 4)
            e._serial = mocker.MagicMock()
            return e

        # Empty data
        enc = make_v1_enc()
        enc._serial.query_struct.return_value = (0,)
        data = enc.get_logged_data()
        assert len(data) == 0
        assert data.dtype == DTYPE_LOGGING_OUT

        # Normal data (no rollover)
        enc = make_v1_enc()
        raw = np.array([(100, 1000), (-200, 2000), (300, 3000)], dtype=raw_dtype)
        enc._serial.query_struct.return_value = (len(raw),)
        enc._serial.read.return_value = raw.tobytes()
        data = enc.get_logged_data()
        assert data.shape == raw.shape
        np.testing.assert_array_equal(
            data['time'], raw['time'].astype('timedelta64[us]')
        )
        np.testing.assert_allclose(
            data['degrees'], raw['tics'] * enc._factor_tic_to_deg
        )

        # Data with two uint32 rollovers (covers both loop branches)
        # time[1] and time[4] simulate a 32-bit microsecond timer wrapping around
        enc = make_v1_enc()
        raw = np.array(
            [(100, 4294967000), (200, 100), (300, 3000), (400, 4294967200), (500, 100)],
            dtype=raw_dtype,
        )
        enc._serial.query_struct.return_value = (len(raw),)
        enc._serial.read.return_value = raw.tobytes()
        data = enc.get_logged_data()
        assert data.shape == raw.shape
        # rollover_indices = [1, 4]; corrections: [1:4] += 2^32, [4:5] += 2*2^32
        expected = raw['time'].astype('int64')
        expected[1:4] += 2**32
        expected[4:5] += 2 * 2**32
        np.testing.assert_array_equal(data['time'].astype('int64'), expected)

    def test_set_stream_prefix(self, mock_encoder, mock_ext_serial, caplog):
        """set_stream_prefix sends a single-byte prefix and rejects invalid inputs."""
        enc = mock_encoder('fake_port')

        # v2 raises immediately
        if enc.hardware_version == 2:
            with pytest.raises(RuntimeError, match='only supported for'):
                enc.set_stream_prefix(b'M')
            return

        # v1: success with bytes
        mock_ext_serial.mock_responses = {b'IM': b'\x01'}
        with caplog.at_level(logging.DEBUG):
            enc.set_stream_prefix(b'M')
        assert 'Setting stream prefix' in caplog.text

        # success with str (auto-encoded)
        enc.set_stream_prefix('M')

        # Invalid type
        with pytest.raises(ValueError, match='str or bytes'):
            enc.set_stream_prefix(123)

        # Too long
        with pytest.raises(ValueError, match='length of 1'):
            enc.set_stream_prefix(b'MB')

        # Failure response
        mock_ext_serial.mock_responses = {b'IM': b'\x00'}
        with pytest.raises(RuntimeError, match='Failed to set stream prefix'):
            enc.set_stream_prefix(b'M')

    def test_set_wrap_mode(self, mock_encoder, mock_ext_serial, caplog):
        """set_wrap_mode sends correct mode byte and updates internal state."""
        enc = mock_encoder('fake_port', encoder_resolution=1024)

        # Invalid mode
        with pytest.raises(ValueError, match='Invalid wrap mode'):
            enc.set_wrap_mode('invalid')

        # bipolar success
        mock_ext_serial.mock_responses = {struct.pack('<cB', b'M', 0): b'\x01'}
        with caplog.at_level(logging.DEBUG):
            enc.set_wrap_mode('bipolar')
        assert 'Setting wrap mode to bipolar' in caplog.text

        # unipolar success
        caplog.clear()
        mock_ext_serial.mock_responses = {struct.pack('<cB', b'M', 1): b'\x01'}
        with caplog.at_level(logging.DEBUG):
            enc.set_wrap_mode('unipolar')
        assert 'Setting wrap mode to unipolar' in caplog.text

        # Failure
        mock_ext_serial.mock_responses = {struct.pack('<cB', b'M', 0): b'\x00'}
        with pytest.raises(RuntimeError, match='Failed to set wrap mode'):
            enc.set_wrap_mode('bipolar')

    def test_set_event_transmission(self, mock_encoder, mock_ext_serial, caplog):
        """set_event_transmission enables/disables threshold event transmission."""
        enc = mock_encoder('fake_port')

        # Enable
        mock_ext_serial.mock_responses = {rb'.*': b'\x01'}
        with caplog.at_level(logging.DEBUG):
            enc.set_event_transmission(True)
        assert enc.event_transmission is True
        assert 'Enabling event transmission' in caplog.text

        # Disable
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            enc.set_event_transmission(False)
        assert enc.event_transmission is False
        assert 'Disabling event transmission' in caplog.text

        # Failure (clear residual buffer bytes before testing the error path)
        mock_ext_serial.response_buffer.clear()
        mock_ext_serial.mock_responses = {rb'.*': b'\x00'}
        with pytest.raises(RuntimeError, match='Failed to'):
            enc.set_event_transmission(True)

    def test_set_usb_stream(self, mock_encoder, mock_ext_serial, mocker):
        """set_usb_stream starts or stops the USB streaming thread as needed."""
        enc = mock_encoder('fake_port')
        mock_ext_serial.mock_responses = {rb'.*': b''}
        mock_thread = mocker.MagicMock()
        enc._usb_stream_thread = mock_thread

        # Enable when not streaming
        mock_thread.is_alive.return_value = False
        enc.set_usb_stream(True)
        mock_thread.start.assert_called_once()

        # Enable when already streaming (no additional start)
        mock_thread.reset_mock()
        mock_thread.is_alive.return_value = True
        enc.set_usb_stream(True)
        mock_thread.start.assert_not_called()

        # Disable when streaming
        mock_thread.reset_mock()
        mock_thread.is_alive.return_value = True
        enc.set_usb_stream(False)
        mock_thread.stop.assert_called_once()

        # Disable when not streaming (no stop)
        mock_thread.reset_mock()
        mock_thread.is_alive.return_value = False
        enc.set_usb_stream(False)
        mock_thread.stop.assert_not_called()
