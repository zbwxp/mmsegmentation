from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .lraspp_head import LRASPPHead
from .nl_head import NLHead
from .ocr_head import OCRHead
# from .point_head import PointHead
# from .psa_head import PSAHead
from .psp_head import PSPHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .uper_head import UPerHead
from .dyn_conv_head import DynConvHead
from .dyn_ppm_head import DynPPMHead
from .deeppad_head import DeepPadHead
from .dyn_aspp_head import DynASPPHead
from .pad_fpn_head import PADFPNHead
from .pad_sem_fpn_head import PADSEMFPNHead
from .deeppad_2block_head import DeepPad2BlockHead
from .deeppad_head_3x3 import DeepPadHead3x3
from .deeppad_head512 import DeepPadHead512
from .bilinear_head import BilinearHead
from .last_hope import LastHopeHead
from .last_hope_v2 import LastHopeHead_v2
from .last_hope_v3 import LastHopeHead_v3
from .last_hope_tower import LastHopeHead_tower
from .last_hope_refine import LastHopeHead_refine
from .baseline_head import BaseHead

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'APCHead', 'DMHead', 'LRASPPHead', 'DynConvHead', 'DynPPMHead', 'DeepPadHead',
    'DynASPPHead', 'PADFPNHead', 'PADSEMFPNHead', 'DeepPad2BlockHead', 'DeepPadHead3x3',
    'DeepPadHead512', 'BilinearHead', 'LastHopeHead', 'LastHopeHead_v2', 'LastHopeHead_v3',
    'LastHopeHead_tower', 'LastHopeHead_refine', 'BaseHead'
]
