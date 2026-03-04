import logging
import struct
from collections.abc import Callable, Sequence
from typing import Literal, cast, overload

import numpy as np
import numpy.typing as npt
from bpod_core.com import ChunkedSerialReader, ExtendedSerial, SerialDevice
from serial.threaded import ReaderThread

log = logging.getLogger(__name__)

DTYPE_LOGGING_IN = np.dtype([('tics', np.int32), ('time', np.uint32)])
DTYPE_LOGGING_OUT = np.dtype([('time', 'timedelta64[us]'), ('degrees', 'f8')])
STRUCT_POSITION = struct.Struct('<xhI')
STRUCT_EVENT = struct.Struct('<x2BI')
MAX_N_THRESHOLDS = 8


class RotaryEncoderModule(SerialDevice):
    def __init__(
        self, port: str, encoder_resolution: int = 1024, reset_to_defaults: bool = True
    ):
        """Create a RotaryEncoderModule instance and open the connection.

        Parameters
        ----------
        port : str
            Serial port name (e.g., ``'/dev/ttyACM0'`` or ``'COM5'``).
        encoder_resolution : int, optional
            The incremental encoder's resolution in pulses per revolution.
            Defaults to 1024.
        reset_to_defaults : bool, optional
            Whether to reset the Rotary Encoder Module to default settings.
            Defaults to True.

        Raises
        ------
        SerialException
            If the port cannot be opened during the probe step.
        ValueError
            If the device on the given port does not appear to be a Rotary
            Encoder Module.
        """
        super().__init__(port=port, open_connection=False)

        # some default attributes
        self._is_sd_logging = False
        self._resolution: int
        self._factor_tic_to_deg: float
        self._factor_deg_to_tic: float
        self._wrap_mode: Literal['bipolar', 'unipolar'] = 'bipolar'
        self._wrap_point_tics: int
        self._thresholds: list[float] = []
        self._event_transmission: bool = False
        self._callback_position: Callable[[int, float], None] | None = None
        self._callback_event: Callable[[int, int, int], None] | None = None

        # handshake / identify hardware version
        self._hardware_version = self.probe(port)
        self._serial_device_name = 'Rotary Encoder Module'
        self.open()

        # rotary encoder module v1 uses X1 encoding, v2 uses X4 encoding
        self._clock_multiplier = 1 if self._hardware_version == 1 else 4

        # set encoder resolution
        self.set_resolution(encoder_resolution)

        # reset to default settings
        if reset_to_defaults:
            self.reset()

        # initialize serial reader thread
        protocol = ChunkedSerialReader(chunk_size=7, callback=self._process_stream)
        self._usb_stream_thread = ReaderThread(self._serial, protocol)

    @property
    def clock_multiplier(self) -> int:
        """Clock multiplier of the Rotary Encoder Module."""
        return self._clock_multiplier

    @property
    def event_transmission(self) -> bool:
        """The state of event transmission."""
        return self._event_transmission

    @property
    def hardware_version(self) -> int:
        """Hardware version of the Rotary Encoder Module."""
        return self._hardware_version

    @property
    def resolution(self) -> int:
        """Resolution of the rotary encoder in pulses per revolution."""
        return self._resolution

    @property
    def sd_logging(self) -> bool:
        """The state of SD card logging."""
        return self._is_sd_logging

    @property
    def thresholds(self) -> list[float]:
        """List of thresholds in degrees."""
        return self._thresholds

    @property
    def is_usb_streaming(self) -> bool:
        """Whether the USB stream is active or not."""
        return self._usb_stream_thread.is_alive()

    @property
    def wrap_mode(self) -> Literal['bipolar', 'unipolar']:
        """The wrap mode of the rotary encoder."""
        return self._wrap_mode

    @property
    def wrap_point(self) -> float:
        """The current wrap point in degrees."""
        return self._tics_to_degrees(self._wrap_point_tics)

    @staticmethod
    @overload
    def probe(port: str, raise_value_error: Literal[True] = True) -> int: ...

    @staticmethod
    @overload
    def probe(port: str, raise_value_error: Literal[False]) -> int | None: ...

    @staticmethod
    def probe(port: str, raise_value_error: bool = True):
        """
        Probe for a Rotary Encoder Module on the specified port.

        Parameters
        ----------
        port : str
            The port to probe.
        raise_value_error : bool, optional
            Whether to raise a ValueError when device does not appear to be a rotary
            encoder. Defaults to True.

        Returns
        -------
        int or None
            The hardware version of the Rotary Encoder Module.
            Will return None if the device does not appear to be a Rotary Encoder Module
            and `raise_value_error` is False.

        Raises
        ------
        SerialException
            If the port cannot be opened.
        ValueError
            If the device does not appear to be a Rotary Encoder Module and
            `raise_value_error` is True.
        """

        def _probe() -> int:
            with ExtendedSerial(port, timeout=0.1) as s:
                s.reset_input_buffer()
                reply = s.query(b'CI\xfa', 2)
            if not reply:
                raise TimeoutError(
                    f'Device on {port} did not respond to query '
                    f'within {s.timeout} seconds.'
                )
            match reply:
                case b'\xd9\x01':
                    return 1
                case b'\xd9\x00':
                    return 2
                case _:
                    raise NotImplementedError(
                        f'Unexpected response from device on {port}: {reply!r}'
                    )

        try:
            return _probe()
        except (TimeoutError, NotImplementedError) as e:
            if raise_value_error:
                raise ValueError(
                    f'Device on {port} does not appear to be a Rotary Encoder Module.'
                ) from e

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.set_sd_logging(False)
        super().__exit__(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        """Close serial connection to the Rotary Encoder Module."""
        self.set_sd_logging(False)
        super().close()

    def _process_stream(self, data: bytes) -> None:
        """
        Process incoming data from the serial reader thread.

        data : bytes
            Incoming data from the serial reader thread.
        """
        match data[0]:
            case 0x50:  # b'P' - position
                if self._callback_position is None:
                    return
                tics, microseconds = STRUCT_POSITION.unpack(data)
                degrees = tics * self._factor_tic_to_deg
                self._callback_position(microseconds, degrees)
            case 0x45:  # b'E' - event
                if self._callback_event is None:
                    return
                event_type, event_code, microseconds = STRUCT_EVENT.unpack(data)
                self._callback_event(event_type, event_code, microseconds)
            case _:  # unknown
                log.error('Unknown message type received: %s', data[0:1])

    def _degrees_to_tics(self, degrees: float) -> int:
        """Convert degrees to tics."""
        return round(degrees * self._factor_deg_to_tic)

    def _tics_to_degrees(self, tics: int) -> float:
        """Convert tics to degrees."""
        return tics * self._factor_tic_to_deg

    def _reset_data_streams(self):
        self._serial.write(b'X')
        self._is_sd_logging = False
        log.debug('All data streams reset')

    def get_tics(self) -> int:
        """
        Get the current encoder position in tics.

        Returns
        -------
        int
            The current encoder position in tics.
        """
        return cast('int', self._serial.query_struct(b'Q', '<h')[0])

    def set_tics(self, tics: int):
        """
        Set the current encoder position in tics.

        Parameters
        ----------
        tics : int
            The new encoder position in tics.
            Value must be an integer between -32768 and 32767.

        Raises
        ------
        RuntimeError
            When setting of the encoder position fails.
        ValueError
            When the new encoder position is outside the allowed range.
        """
        try:
            if not self._serial.verify(struct.pack('<ch', b'P', tics)):
                raise RuntimeError('Failed to set encoder position')
        except struct.error as e:
            tics_min = np.iinfo(np.int16).min
            tics_max = np.iinfo(np.int16).max
            raise ValueError(
                f'New encoder position must be between {tics_min} and {tics_max} tics'
            ) from e

    def get_degrees(self) -> float:
        """
        Get the current encoder position in degrees.

        Returns
        -------
        float
            The current encoder position in degrees.
        """
        return self._tics_to_degrees(self.get_tics())

    def set_degrees(self, degrees: float):
        """
        Set the current encoder position in degrees.

        Parameters
        ----------
        degrees : float
            The new encoder position in degrees.

        Raises
        ------
        RuntimeError
            When setting of the encoder position fails.
        ValueError
            When the new encoder position is outside the allowed range.
        """
        try:
            tics = self._degrees_to_tics(degrees)
            self.set_tics(tics)
            if log.isEnabledFor(logging.DEBUG):
                degrees = self._tics_to_degrees(tics)
                log.debug('Setting encoder position to %0.1f°', degrees)
        except ValueError as e:
            degs_min = self._tics_to_degrees(np.iinfo(np.int16).min)
            degs_max = self._tics_to_degrees(np.iinfo(np.int16).max)
            raise ValueError(
                f'New encoder position must be between '
                f'{degs_min:.1f}° and {degs_max:.1f}°.'
            ) from e

    def reset(self):
        """Reset Rotary Encoder Module to default settings."""
        self.set_wrap_point(180.0)
        self.set_thresholds([-40.0, 40.0])
        self.set_wrap_mode('bipolar')
        self.set_event_transmission(False)
        # obj.moduleOutputStream = 'off';
        if self._hardware_version == 1:
            self.set_stream_prefix(b'M')

    def set_resolution(self, value: int) -> None:
        """
        Set the encoder resolution in tics per revolution.

        Parameters
        ----------
        value : int
            The new encoder resolution in tics per revolution.
            Must be a positive integer.

        Raises
        ------
        ValueError
            If the encoder resolution is not a positive integer.
        """
        if value <= 0:
            raise ValueError('Encoder resolution must be a positive integer.')
        self._resolution = int(value)
        self._factor_deg_to_tic = self._resolution * self._clock_multiplier / 360.0
        self._factor_tic_to_deg = 1 / self._factor_deg_to_tic

    def set_wrap_point(self, degrees: float) -> None:
        """
        Set the wrap point in degrees.

        Parameters
        ----------
        degrees : float
            The new wrap point in degrees.
        """
        tics = self._degrees_to_tics(abs(degrees))
        query = struct.pack('<ch', b'W', tics)
        if self._serial.verify(query):
            self._wrap_point_tics = tics
            if log.isEnabledFor(logging.DEBUG):
                log.debug('Setting wrap point to %0.1f°', self.wrap_point)
        else:
            raise RuntimeError('Failed to set wrap point')

    def set_thresholds(self, degrees: Sequence[float]) -> None:
        """
        Set the thresholds in degrees.

        Parameters
        ----------
        degrees : Sequence of float
            Thresholds in degrees. The sequence must not contain more than 8 values.

        Raises
        ------
        ValueError
            If any threshold exceeds the current wrap point or the number of thresholds
            exceeds 8.
        RuntimeError
            If setting of the thresholds fails.
        """
        wrap_point = self.wrap_point  # avoid unnecessary computation
        if (n_thresholds := len(degrees)) > MAX_N_THRESHOLDS:
            raise ValueError(f'A maximum of {MAX_N_THRESHOLDS} thresholds can be set.')
        if any(abs(threshold) > wrap_point for threshold in degrees):
            raise ValueError(
                f'Threshold values cannot exceed the current wrap point of '
                f'{wrap_point}°.'
            )
        tics = [self._degrees_to_tics(thresh) for thresh in degrees]
        actual_degrees = [self._tics_to_degrees(tick) for tick in tics]
        query = struct.pack(f'<cB{n_thresholds}h', b'T', n_thresholds, *tics)
        if self._serial.verify(query):
            self._thresholds = actual_degrees
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    'Setting thresholds to %s',
                    ', '.join([f'{x:0.1f}°' for x in actual_degrees]),
                )
        else:
            raise RuntimeError('Failed to set thresholds')

    def set_sd_logging(self, enable_logging: bool) -> None:
        """
        Enable or disable SD card logging.

        Parameters
        ----------
        enable_logging : bool
            Whether to enable SD card logging. True enables logging, False disables it.

        Raises
        ------
        RuntimeError
            If the hardware does not support SD card logging.
        """
        if enable_logging == self._is_sd_logging:
            return
        if self.hardware_version != 1:
            raise RuntimeError(
                f'SD card logging is not supported on {self._serial_device_name} '
                f'v{self.hardware_version}'
            )
        if enable_logging:
            self._serial.write(b'L')
            log.debug('SD Logging enabled')
        else:
            self._serial.write(b'F')
            log.debug('SD Logging disabled')
        self._is_sd_logging = bool(enable_logging)

    def zero(self) -> None:
        """Reset current encoder position to zero."""
        log.debug('Resetting encoder position to 0°.')
        self._serial.write(b'Z')

    def get_logged_data(self) -> npt.NDArray[np.void]:
        """Retrieve logged data from the SD card.

        Returns
        -------
        numpy.ndarray
            Structured NumPy array with one element per logged sample.
            Each element has the following fields:

            - ``time`` :class:`numpy.timedelta64` with unit ``us``
            - ``degrees`` :class:`numpy.float64`
        """
        if self.hardware_version != 1:
            raise RuntimeError(
                f'SD card logging is not supported on {self._serial_device_name} '
                f'v{self.hardware_version}'
            )

        self.set_sd_logging(False)  # stop logging before retrieving data

        # prepare output array
        n_records = self._serial.query_struct(b'R', '<I')[0]
        out = np.empty(n_records, dtype=DTYPE_LOGGING_OUT)
        if n_records == 0:
            return out

        # retrieve data from rotary encoder module and parse into structured array
        buffer = self._serial.read(n_records * 8)
        raw_data = np.frombuffer(buffer, dtype=DTYPE_LOGGING_IN)
        out['time'] = raw_data['time'].astype('timedelta64[us]')
        np.multiply(raw_data['tics'], self._factor_tic_to_deg, out=out['degrees'])

        # Correct rollover in 32-bit microsecond timer
        time_signed = raw_data['time'].astype(np.int64)
        rollover_indices = np.where(np.diff(time_signed) < 0)[0] + 1
        if rollover_indices.size:
            for i, start in enumerate(rollover_indices):
                end = (
                    rollover_indices[i + 1]
                    if i + 1 < len(rollover_indices)
                    else n_records
                )
                delta = np.timedelta64((i + 1) * 2**32, 'us')
                out['time'][start:end] += delta

        return out

    def set_stream_prefix(self, prefix: str | bytes = b'M') -> None:
        """
        Set the stream prefix.

        Parameters
        ----------
        prefix : str or bytes, optional
            A single character or byte to be used as the stream prefix.

        Raises
        ------
        ValueError
            If the prefix is not a single character or byte.
        RuntimeError
            If the hardware version is not 1 or if setting the prefix fails.
        """
        # Raise exception if not version 1
        if self.hardware_version != 1:
            raise RuntimeError(
                f'Setting of stream prefix is only supported for '
                f'{self._serial_device_name} v1'
            )

        # validate prefix and convert to bytes if necessary
        match prefix:
            case str():
                prefix = prefix.encode()
            case bytes():
                pass
            case _:
                raise ValueError('Stream prefix must be of type str or bytes.')
        if len(prefix) > 1:
            raise ValueError('Stream prefix must have a length of 1.')

        # send command and read response
        if self._serial.verify(b'I' + prefix):
            log.debug('Setting stream prefix to %s', prefix)
        else:
            raise RuntimeError('Failed to set stream prefix')

    def set_wrap_mode(self, mode: Literal['bipolar', 'unipolar']):
        if mode not in ('unipolar', 'bipolar'):
            raise ValueError(
                'Invalid wrap mode. Must be either "bipolar" or "unipolar".'
            )
        self._serial.write_struct('<cB', b'M', 0 if mode == 'bipolar' else 1)
        if self._serial.read() == b'\x01':
            log.debug('Setting wrap mode to %s', mode)
            self._wrap_mode = mode
        else:
            raise RuntimeError(f'Failed to set wrap mode to {mode}')

    def set_event_transmission(self, value: bool):
        self._serial.write_struct('<c?', b'V', bool(value))
        if self._serial.verify():
            log.debug('%sabling event transmission', 'En' if value else 'Dis')
            self._event_transmission = value
        else:
            raise RuntimeError(
                f'Failed to {"en" if value else "dis"}able event transmission'
            )

    def enable_thresholds(self, value: bool | str | Sequence[int]):
        """
        Enable or disable thresholds based on the provided value.

        Parameters
        ----------
        value : bool, str or Sequence of int
            - If `bool`: enables all 8 thresholds if True, or disables them if False.
            - If `str`: must be a binary string of exactly 8 characters
              (e.g., '11010100'). Each character represents whether the corresponding
              threshold (bit 0 to 7) should be enabled (1) or disabled (0).
            - If `Sequence[int]`: a sequence of integers (ranging from 0 to 7)
              specifying which thresholds to enable. All other thresholds will be
              disabled.

        Raises
        ------
        ValueError
            If `value` is a string that is not a valid 8-character binary string, or a
            sequence containing integers outside the range 0 to 7.
        TypeError
            If `value` is not a bool, str, or sequence of int.
        """
        if isinstance(value, bool):
            byte_value = 0xFF if value else 0x00
        elif isinstance(value, str):
            if len(value) == 8 and all(c in '01' for c in value):
                byte_value = int(value, 2)
            else:
                raise ValueError("String must be 8 characters of '0' or '1'.")
        elif isinstance(value, Sequence):
            byte_value = 0
            for x in value:
                if not (isinstance(x, int) and 0 <= x < 8):
                    raise ValueError('Sequence must contain integers in range 0 to 7.')
                byte_value |= 1 << x

        else:
            raise TypeError('Unsupported input type.')

        self._serial.write_struct('<cB', b';', byte_value)

        if log.isEnabledFor(logging.DEBUG):
            if byte_value == 0xFF:
                log.debug('Enabled all 8 thresholds')
            elif byte_value == 0x00:
                log.debug('Disabled all 8 thresholds')
            else:
                enabled = [str(x) for x in range(8) if (byte_value & (1 << x)) != 0]
                disabled = [str(x) for x in range(8) if (byte_value & (1 << x)) == 0]
                log.debug(
                    'Enabled threshold%s %s; disabled threshold%s %s',
                    's' if len(enabled) > 1 else '',
                    ', '.join(enabled),
                    's' if len(disabled) > 1 else '',
                    ', '.join(disabled),
                )

    def set_usb_stream(self, enable: bool) -> None:
        self._serial.write_struct('<c?', b'S', bool(enable))
        if enable and not self.is_usb_streaming:
            self._usb_stream_thread.start()
        elif not enable and self.is_usb_streaming:
            self._usb_stream_thread.stop()

    def set_position_callback(
        self, callback_function: Callable[[int, float], None]
    ) -> None:
        self._callback_position = callback_function

    def set_event_callback(
        self, callback_function: Callable[[int, int, int], None]
    ) -> None:
        self._callback_event = callback_function
