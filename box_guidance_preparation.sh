#!/bin/bash
cd Data/
git clone https://huggingface.co/datasets/YangCaoCS/codav2_box_guidance
mv codav2_box_guidance/scannet_groundingDino_2dboxes.tar scannet/
cd scannet/
tar -xvf scannet_groundingDino_2dboxes.tar
cd -
mv codav2_box_guidance/sunrgbd_groundingDino_2dboxes.tar sunrgb_d/
cd sunrgb_d/
tar -xvf sunrgbd_groundingDino_2dboxes.tar
cd -
echo "Finished"
