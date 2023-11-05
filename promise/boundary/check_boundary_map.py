import nibabel as nib
import torch.nn as nn
import torch
"""
check the boundary map
"""
seg_data = nib.load('')
seg = seg_data.get_fdata()
seg = torch.from_numpy(seg).float().cuda().unsqueeze(0)

m_xy = nn.AvgPool3d((5,5,1), stride=1, padding=(2,2,0)).cuda()

# can do any view
# m_yz = nn.AvgPool3d((1,5,5), stride=1, padding=(0,2,2)).cuda()
# m_xz = nn.AvgPool3d((5,1,5), stride=1, padding=(2,0,2)).cuda()

edge_map = abs(seg - m_xy(seg))[0, :]  # edge map can be easily obtained from any mask
edge_image = nib.Nifti1Image(edge_map.cpu().numpy(), seg_data.affine, seg_data.header)
nib.save(edge_image, '')


