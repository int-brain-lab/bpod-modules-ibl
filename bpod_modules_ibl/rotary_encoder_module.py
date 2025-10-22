import logging
import struct
from collections.abc import Sequence
from typing import Literal, cast, overload

import numpy as np
from bpod_core.com import ExtendedSerial
from numpy.typing import NDArray
from serial import SerialException

log = logging.getLogger(__name__)

DTYPE_LOGGING = np.dtype([('time', 'timedelta64[us]'), ('degrees', 'f8')])


class RotaryEncoderModule:
    _name: str = 'Rotary Encoder Module'
    _is_sd_logging: bool = False
    _resolution: int = 1024
    _clock_multiplier: int
    _factor_tic_to_deg: float
    _factor_deg_to_tic: float
    _wrap_mode: Literal['bipolar', 'unipolar'] = 'bipolar'
    _wrap_point_tics: int
    _thresholds: list[float] = []
    _max_thresholds: int = 8
    _event_transmission: bool = False

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
        # handshake / identify hardware version
        self._hardware_version = self.probe(port)

        # rotary encoder module v1 uses X1 encoding, v2 uses X4 encoding
        self._clock_multiplier = 1 if self._hardware_version == 1 else 4

        # set encoder resolution
        self.set_resolution(encoder_resolution)

        # initialize serial object and set port
        # implemented that awkwardly to get logging from self.open()
        self._serial = ExtendedSerial()
        self._serial.port = port
        self.open()

        # reset to default settings
        if reset_to_defaults:
            self.reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

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
    def port(self) -> str | None:
        """Port name of the Rotary Encoder Module."""
        return self._serial.port

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
        hardware_version = None
        try:
            with ExtendedSerial(port, timeout=0.1) as s:
                s.reset_input_buffer()
                reply = s.query(b'CI\xfa', 2)
            if len(reply) == 0:
                raise TimeoutError(
                    f'Device on {port} did not respond to query within {s.timeout} '
                    f'seconds.'
                )
            match reply:
                case b'\xd9\x01':
                    hardware_version = 1
                case b'\xd9\x00':
                    hardware_version = 2
                case _:
                    raise NotImplementedError(
                        f'Unexpected response from device on {port}: {reply!r}'
                    )
        except SerialException as e:
            if 'could not open port' in str(e):
                raise SerialException(
                    f'Could not connect to device on {port}. Is the device connected?'
                ) from e
            else:
                raise
        except (TimeoutError, NotImplementedError) as e:
            if raise_value_error:
                raise ValueError(
                    f'Device on {port} does not appear to be a Rotary Encoder Module.'
                ) from e
        return hardware_version

    def open(self) -> None:
        """Open serial connection to the Rotary Encoder Module."""
        if not self._serial.is_open:
            log.debug(
                'Opening serial connection to %s v%d on %s',
                self._name,
                self._hardware_version,
                self.port,
            )
            self._serial.open()

    def close(self) -> None:
        """Close serial connection to the Rotary Encoder Module."""
        self.set_sd_logging(False)
        if hasattr(self, '_serial') and self._serial.is_open:
            log.debug(
                'Closing serial connection to %s v%d on %s',
                self._name,
                self._hardware_version,
                self.port,
            )
            self._serial.close()

    def _degrees_to_tics(self, degrees: float) -> int:
        """Convert degrees to tics."""
        return int(round(degrees * self._factor_deg_to_tic))

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
        if any(abs(threshold) > self.wrap_point for threshold in degrees):
            raise ValueError(
                f'Threshold values cannot exceed the current wrap point of '
                f'{self.wrap_point}°.'
            )
        if (n_thresholds := len(degrees)) > 8:
            raise ValueError(
                f'A maximum of {self._max_thresholds} thresholds can be set.'
            )
        tics = [self._degrees_to_tics(thresh) for thresh in degrees]
        degrees = [self._tics_to_degrees(tick) for tick in tics]
        query = struct.pack(f'<cB{n_thresholds}h', b'T', n_thresholds, *tics)
        if self._serial.verify(query):
            self._thresholds = degrees
            log.debug(
                'Setting thresholds to %s', ', '.join([f'{x:0.1f}°' for x in degrees])
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
                f'SD card logging is not supported on {self._name} '
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

    def get_logged_data(self) -> NDArray[np.void]:
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
                f'SD card logging is not supported on {self._name} '
                f'v{self.hardware_version}'
            )

        self.set_sd_logging(False)  # stop logging before retrieving data

        # prepare output array
        n_records = self._serial.query_struct(b'R', '<I')[0]
        out = np.empty(n_records, dtype=DTYPE_LOGGING)
        if n_records == 0:
            return out

        # retrieve data from rotary encoder module and parse into structured array
        buffer = self._serial.read(n_records * 8)
        dtype = np.dtype([('tics', np.int32), ('time', np.uint32)])
        raw_data = np.frombuffer(buffer, dtype=dtype)
        out['time'] = raw_data['time'].astype('timedelta64[us]')
        np.multiply(raw_data['tics'], self._factor_tic_to_deg, out=out['degrees'])

        # Correct rollover in 32-bit microsecond timer
        rollover_indices = np.where(np.diff(raw_data['time']) < 0)[0] + 1
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
                f'Setting of stream prefix is only supported for {self._name} v1'
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
        if mode not in ['bipolar', 'unipolar']:
            raise ValueError(
                'Invalid wrap mode. Must be either "bipolar" or "unipolar".'
            )
        self._serial.write_struct('<cB', b'M', 0 if mode == 'bipolar' else 1)
        if self._serial.read() == b'\x01':
            log.debug('Setting wrap mode to %s', mode)
        else:
            raise RuntimeError(f'Failed to set wrap mode to {mode}')

    def set_event_transmission(self, value: bool):
        self._serial.write_struct('<c?', b'V', bool(value))
        if self._serial.verify(b''):
            log.debug('%sabling event transmission', 'En' if value else 'Dis')
        else:
            raise RuntimeError(
                f'Failed to {"en" if value else "dis"}able event transmission'
            )

    def enable_thresholds(self, value: bool | str | Sequence[int]):
        """
        Enable or disable thresholds based on the provided value.

        Parameters
        ----------
        value : bool or str or Sequence of int
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
            If `value` is not a bool, a valid binary string, or a valid sequence of
            integers.
        """
        byte_value = 0
        if isinstance(value, bool):
            byte_value = 0xFF if value else 0x00
        elif isinstance(value, str):
            if len(value) == 8 and all(c in '01' for c in value):
                byte_value = int(value, 2)
            else:
                raise ValueError("String must be 8 characters of '0' or '1'.")
        elif isinstance(value, Sequence) and not isinstance(value, str):
            if all(isinstance(x, int) and 0 <= x < 8 for x in value):
                byte_value = 0
                for bit in value:
                    byte_value |= 1 << bit
            else:
                raise ValueError('Sequence must contain integers in range 0 to 7.')
        else:
            raise ValueError("Unsupported input type for 'value'.")
        self._serial.write_struct('<cB', b';', byte_value)

    def enable_evt_transmission(self):
        pass
