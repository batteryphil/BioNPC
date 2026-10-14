"""
Optional zero-overhead Python ctypes binding to `libbionpc.so`.
Allows Python applications and test suites to invoke the native C inference engine
directly for bit-exact cross-language verification and lightweight CPU deployment.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

BIONPC_MAX_LAYERS = 16
BIONPC_MAX_DIM = 64


class _CConfig(ctypes.Structure):
    _fields_ = [
        ("d_in", ctypes.c_int),
        ("d_out", ctypes.c_int),
        ("d_model", ctypes.c_int),
        ("n_layers", ctypes.c_int),
    ]


class _CAgentState(ctypes.Structure):
    _fields_ = [
        ("apl_state", ctypes.c_float * BIONPC_MAX_LAYERS),
        ("apl_locked", ctypes.c_ubyte * BIONPC_MAX_LAYERS),
    ]


class _CControl(ctypes.Structure):
    _fields_ = [
        ("sensory_scale", ctypes.c_float * BIONPC_MAX_DIM),
        ("sensory_bias", ctypes.c_float * BIONPC_MAX_DIM),
        ("apl_min", ctypes.c_float),
        ("apl_max", ctypes.c_float),
        ("apl_locked", ctypes.c_int),
        ("chemical_bias", ctypes.c_float),
        ("action_scale", ctypes.c_float * BIONPC_MAX_DIM),
        ("action_min", ctypes.c_float * BIONPC_MAX_DIM),
        ("action_max", ctypes.c_float * BIONPC_MAX_DIM),
        ("action_override_mask", ctypes.c_ubyte * BIONPC_MAX_DIM),
        ("action_override_val", ctypes.c_float * BIONPC_MAX_DIM),
        ("archetype_name", ctypes.c_char * 64),
    ]


class NativeBioNPC:
    """Wrapper around native C `libbionpc.so`."""

    def __init__(self, weights_path: Union[str, Path], lib_path: Optional[Union[str, Path]] = None):
        if lib_path is None:
            lib_path = Path(__file__).resolve().parent.parent.parent / "c" / "lib" / "libbionpc.so"
        self.lib = ctypes.CDLL(str(lib_path))

        self.lib.bionpc_model_load_file.argtypes = [ctypes.c_char_p]
        self.lib.bionpc_model_load_file.restype = ctypes.c_void_p

        self.lib.bionpc_model_free.argtypes = [ctypes.c_void_p]
        self.lib.bionpc_model_free.restype = None

        self.lib.bionpc_forward_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(_CAgentState),
            ctypes.c_int,
        ]
        self.lib.bionpc_forward_batch.restype = None

        self.lib.bionpc_crc32_states.argtypes = [
            ctypes.POINTER(_CAgentState),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.bionpc_crc32_states.restype = ctypes.c_uint32

        self._handle = self.lib.bionpc_model_load_file(str(weights_path).encode("utf-8"))
        if not self._handle:
            raise RuntimeError(f"Failed to load native BioNPC model from {weights_path}")

        # Read config header at start of struct
        cfg = _CConfig.from_address(self._handle)
        self.d_in = cfg.d_in
        self.d_out = cfg.d_out
        self.d_model = cfg.d_model
        self.n_layers = cfg.n_layers

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.lib.bionpc_model_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def forward_batch(
        self,
        inputs: np.ndarray,
        states: Optional[ctypes.Array] = None,
    ) -> Tuple[np.ndarray, ctypes.Array]:
        inputs_f32 = np.ascontiguousarray(inputs, dtype=np.float32)
        if inputs_f32.ndim == 1:
            inputs_f32 = inputs_f32.reshape(1, -1)
        batch_size = inputs_f32.shape[0]
        outputs = np.zeros((batch_size, self.d_out), dtype=np.float32)

        if states is None:
            states = (_CAgentState * batch_size)()

        in_ptr = inputs_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.bionpc_forward_batch(self._handle, in_ptr, out_ptr, states, batch_size)
        return outputs, states

    def crc32_states(self, states: ctypes.Array, batch_size: int) -> int:
        return int(self.lib.bionpc_crc32_states(states, batch_size, self.n_layers))
