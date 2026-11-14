# Third-party notices and scope
|Files|Origin / permission|Release status|
|code/models/RHANet_dep/rhanet_dep.py and modules/rhanet_modules_depthwise.py; inherited package files|RHACrackNet repository MIT notice, Copyright (c) 2023 lovelyyoshino; see third_party/RHACrackNet_LICENSE|Retain notice; do not claim original authorship|
|code/models/RHANet_dep/modules/dyrelu.py|Matches https://github.com/Islanna/DynamicReLU/blob/c9326bd2c134793264d2f5d398c78a78783e00bd/dyrelu.py after CRLF/LF normalization; no explicit upstream redistribution license verified (rechecked 2026-11-15)|LICENSE_UNRESOLVED; blocks public distribution|
|code/losses/cross_entropy.py|Inherited RHACrackNet weighted loss implementation|Original MIT attribution retained, not newly authored|
|code/models/cssg/; code/models/RHANet_CSSG/; code/models/RHANet_CSSG_LBAS/|PaveCrackNet additions, extending inherited model|User-selected MIT for authorized original portions; copyright-holder identity pending|
|code/dataset/; formal training entries; code/evaluation_utils.py|Project adapters/helpers; imports and calling patterns include inherited framework/model interfaces|Only authorized original portions covered by user's MIT selection; inherited portions retain notices|
|scripts/; configs/; project documentation|Release tooling and original contributions|MIT selection confirmed; final copyright line pending|
|splits/pycrackdb/|Relative metadata from PY-CrackDB source, DOI10.5281/zenodo.16749554, locally recorded CC-BY-4.0|Source attribution in docs/DATASETS.md; no images or masks|

No DeepCrack, Crackmer, PIDNet, BiSeNetV2 or SegFormer sources or weights are included. Do not interpret the root license status as a blanket grant covering unresolved third-party code. Upstream author identity beyond repository owner has not been independently established; no copyright holder is invented.

Upstream recheck: master commit `0ba5a33821024d5e7d098f2343e3bb10a4fe4bb9` and the source commit above contain only README.md and dyrelu.py; no LICENSE file was found and GitHub license metadata is null. Upstream dyrelu.py has no embedded copyright/license notice. The inherited RHACrackNet MIT notice is retained, but does not independently establish permission for this separately sourced component. No new DyReLU code copy is introduced by this documentation-only update; existing public redistribution remains unresolved.
