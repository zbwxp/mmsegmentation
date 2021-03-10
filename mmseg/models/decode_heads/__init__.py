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
from .baseline_aspp_head import BaseASPPHead
from .basepadnet_head import BasePadHead
from .baseASPPpadnet_head import BaseASPPPadHead
from .bilinear_pad_head import BilinearPADHead
from .bilinear_pad_tower_head import BilinearPADTowerHead
from .bilinear_pad_head_v3 import BilinearPADHead_v3
from .Torchseg_base import TorchsegBaseHead
from .stage2_pad_head import Stage2PADHead
from .stage2_pad_head_v3 import Stage2PADHead_v3
from .Torchseg_base_r50 import TorchsegBaseHead_r50
from .Torchseg_base_r50sepbottleneck import TorchsegBaseHead_r50sepbottleneck
from .stage2_pad_head_r50 import Stage2PADHead_r50
from .bilinear_head_noASPP import BilinearHead_noASPP
# from .stage2_ASPP_pad_head_r50 import Stage2PADHeadASPP_r50
from .stage2_pad_head_r50_gp import Stage2PADHead_r50_GP
from .stage2_pad_head_r50_gp_enhance import Stage2PADHead_r50_GPEnhance
from .stage2_pad_head_r50_fast import Stage2PADHead_r50_fast
from .stage2_pad_head_r50_dr1 import Stage2PADHead_r50_dr1
from .stage2_pad_head_r50_gp_fast import Stage2PADHead_r50_GP_fast
from .bilinear_pad_head_fast import BilinearPADHead_fast
from .bilinear_pad_head_fast_no_bottle import BilinearPADHead_fast_no_bottle
from .last_hope_fast import LastHopeHead_fast
from .dyn_conv_head_fast import DynConvHead_fast
from .stage2_pad_head_r50_ASPP_ver import Stage2PADHead_r50_ASPP_ver


__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'APCHead', 'DMHead', 'LRASPPHead', 'DynConvHead', 'DynPPMHead', 'DeepPadHead',
    'DynASPPHead', 'PADFPNHead', 'PADSEMFPNHead', 'DeepPad2BlockHead', 'DeepPadHead3x3',
    'DeepPadHead512', 'BilinearHead', 'LastHopeHead', 'LastHopeHead_v2', 'LastHopeHead_v3',
    'LastHopeHead_tower', 'LastHopeHead_refine', 'BaseHead', 'BaseASPPHead', 'BasePadHead',
    'BaseASPPPadHead', 'BilinearPADHead', 'BilinearPADTowerHead', 'BilinearPADHead_v3',
    'TorchsegBaseHead', 'Stage2PADHead', 'Stage2PADHead_v3', 'TorchsegBaseHead_r50',
    'TorchsegBaseHead_r50sepbottleneck', 'Stage2PADHead_r50', 'BilinearHead_noASPP',
    'Stage2PADHead_r50_GP', 'Stage2PADHead_r50_GPEnhance', 'Stage2PADHead_r50_fast',
    'Stage2PADHead_r50_dr1', 'Stage2PADHead_r50_GP_fast', 'BilinearPADHead_fast',
    'BilinearPADHead_fast_no_bottle', 'LastHopeHead_fast', 'DynConvHead_fast',
    'Stage2PADHead_r50_ASPP_ver'
]
