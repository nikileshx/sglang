# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="stage-a-test-cpu")


def _make_autoround_config(packing_format, weight_bits=4, group_size=128, sym=True):
    return AutoRoundConfig(
        weight_bits=weight_bits,
        group_size=group_size,
        sym=sym,
        packing_format=packing_format,
        block_name_to_quantize=None,
        extra_config=None,
        data_type="int",
        backend="auto",
    )


class TestAutoRoundGetQuantMethod(CustomTestCase):
    """Unit tests for AutoRoundConfig.apply_gptq_quant_layer.

    Verifies that FusedMoE and LinearBase layers get a valid quant method in
    both marlin and non-marlin paths.
    """

    def _make_mock_fused_moe(self):
        """Create a mock that passes isinstance(layer, FusedMoE) checks."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        mock = MagicMock(spec=FusedMoE)
        mock.__class__ = FusedMoE
        return mock

    def _make_mock_linear(self):
        """Create a mock that passes isinstance(layer, LinearBase) checks."""
        from sglang.srt.layers.linear import LinearBase

        mock = MagicMock(spec=LinearBase)
        mock.__class__ = LinearBase
        return mock

    # ------------------------------------------------------------------ #
    # GPTQ tests
    # ------------------------------------------------------------------ #

    @patch(
        "sglang.srt.layers.quantization.marlin_utils.check_moe_marlin_supports_layer",
        return_value=False,
    )
    @patch(
        "sglang.srt.layers.quantization.marlin_utils.check_marlin_supported",
        return_value=False,
    )
    def test_gptq_fused_moe_non_marlin_returns_valid_method(
        self, _mock_marlin_supported, _mock_moe_marlin
    ):
        """Non-marlin GPTQ + FusedMoE must not raise UnboundLocalError."""
        config = _make_autoround_config("auto_round:auto_gptq")
        layer = self._make_mock_fused_moe()

        # This used to raise UnboundLocalError: 'GPTQMarlinMoEMethod'
        result = config.apply_gptq_quant_layer(layer, prefix="model.layers.0.moe")
        self.assertIsNotNone(result)

    @patch(
        "sglang.srt.layers.quantization.marlin_utils.check_moe_marlin_supports_layer",
        return_value=True,
    )
    @patch(
        "sglang.srt.layers.quantization.marlin_utils.check_marlin_supported",
        return_value=True,
    )
    def test_gptq_fused_moe_marlin_returns_marlin_method(
        self, _mock_marlin_supported, _mock_moe_marlin
    ):
        """Marlin GPTQ + FusedMoE should return GPTQMarlinMoEMethod."""
        from sglang.srt.layers.quantization.gptq import GPTQMarlinMoEMethod

        config = _make_autoround_config("auto_round:auto_gptq")
        layer = self._make_mock_fused_moe()

        result = config.apply_gptq_quant_layer(layer, prefix="model.layers.0.moe")
        self.assertIsInstance(result, GPTQMarlinMoEMethod)

    @patch(
        "sglang.srt.layers.quantization.marlin_utils.check_marlin_supported",
        return_value=False,
    )
    def test_gptq_linear_non_marlin_returns_gptq_method(self, _mock_marlin_supported):
        """Non-marlin GPTQ + LinearBase should return GPTQLinearMethod."""
        from sglang.srt.layers.quantization.gptq import GPTQLinearMethod

        config = _make_autoround_config("auto_round:auto_gptq")
        layer = self._make_mock_linear()

        result = config.apply_gptq_quant_layer(layer, prefix="model.layers.0.linear")
        self.assertIsInstance(result, GPTQLinearMethod)

    @patch(
        "sglang.srt.layers.quantization.marlin_utils.check_marlin_supported",
        return_value=True,
    )
    def test_gptq_linear_marlin_returns_marlin_method(self, _mock_marlin_supported):
        """Marlin GPTQ + LinearBase should return GPTQMarlinLinearMethod."""
        from sglang.srt.layers.quantization.gptq import GPTQMarlinLinearMethod

        config = _make_autoround_config("auto_round:auto_gptq")
        layer = self._make_mock_linear()

        result = config.apply_gptq_quant_layer(layer, prefix="model.layers.0.linear")
        self.assertIsInstance(result, GPTQMarlinLinearMethod)


if __name__ == "__main__":
    unittest.main()
