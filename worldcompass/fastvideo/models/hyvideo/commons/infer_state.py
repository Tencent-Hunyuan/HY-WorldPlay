# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from typing import Optional
from dataclasses import dataclass


@dataclass
class InferState:
    enable_sageattn: bool = False  # whether to use SageAttention
    sage_blocks_range: Optional[range] = (
        None  # block range to use SageAttention
    )
    enable_torch_compile: bool = False  # whether to use torch compile


__infer_state = None


def parse_range(value):
    if "-" in value:
        start, end = map(int, value.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(x) for x in value.split(",")]


def initialize_infer_state(args):
    global __infer_state
    sage_blocks_range = parse_range(args.sage_blocks_range)
    # Map CLI argument use_sageattn to internal enable_sageattn field
    use_sageattn = getattr(args, "use_sageattn", False)
    __infer_state = InferState(
        enable_sageattn=use_sageattn,
        sage_blocks_range=sage_blocks_range,
        enable_torch_compile=args.enable_torch_compile,
    )
    return __infer_state


def get_infer_state():
    return __infer_state
