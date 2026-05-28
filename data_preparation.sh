#!/bin/bash
mv sunrgbd_trainval.tar Data/sunrgb_d/
mv sunrgbd_v1_revised_0415.tar Data/sunrgb_d/
cd Data/sunrgb_d/
tar -xvf sunrgbd_v1_revised_0415.tar
tar -xvf sunrgbd_trainval.tar
cd -
mv scannet200_data.tar.* Data/scannet/
cd Data/scannet/
cat scannet200_data.tar.* | tar -xvf
cd -
echo "Finished"