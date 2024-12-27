#!/bin/bash

# 下载SUNRBGD文件的 URL
url="https://hkustconnect-my.sharepoint.com/personal/ycaobd_connect_ust_hk/_layouts/15/download.aspx?UniqueId=27fbe743-303a-480b-8ab7-eac0d859f2ca&Translate=false&tempauth=v1.eyJzaXRlaWQiOiJlZjc3ODAyMC0yOTc1LTQ0ZDYtOGFiMC1iYTA1ZjU1ZTViODkiLCJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvaGt1c3Rjb25uZWN0LW15LnNoYXJlcG9pbnQuY29tQDZjMWQ0MTUyLTM5ZDAtNDRjYS04OGQ5LWI4ZDZkZGNhMDcwOCIsImV4cCI6IjE3MzUzMzMzODUifQ.CgkKBHNuaWQSATgSCwjipfKbs6TTPRAFGg00NS4xNDYuMjMyLjc4IhRtaWNyb3NvZnQuc2hhcmVwb2ludCosd0tVR05lVDgzOHQ2WkwwbjB0L1BlSmtHTithWFRMdVpEZWU5Q3k3dUpiZz0wnQE4AUIQoXGr0m6wAEBJnfDXYN_1CEoQaGFzaGVkcHJvb2Z0b2tlbmIEdHJ1ZWokOWNiNzcyZWEtYTk1OS00ZGM1LWJhODEtOTBkYTAzNGJkYzhjcikwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDFjOTFiNTI4NkBsaXZlLmNvbXoBMMIBJTAjLmZ8bWVtYmVyc2hpcHx5Y2FvYmRAY29ubmVjdC51c3QuaGvIAQE.-ORn6qmURj-C_KJmYuivho0itm5NqefbX3qEqodLt44"

# 输出文件名
output_file="sunrgbd_trainval.tar"

# 使用 curl 下载文件
curl -L -o "$output_file" "$url"

echo "下载完成：$output_file"


# 下载SUNRBGD文件的 URL
url="https://hkustconnect-my.sharepoint.com/personal/ycaobd_connect_ust_hk/_layouts/15/download.aspx?UniqueId=c1818df7-711a-4bf4-81ea-97e964770a0d&Translate=false&tempauth=v1.eyJzaXRlaWQiOiJlZjc3ODAyMC0yOTc1LTQ0ZDYtOGFiMC1iYTA1ZjU1ZTViODkiLCJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvaGt1c3Rjb25uZWN0LW15LnNoYXJlcG9pbnQuY29tQDZjMWQ0MTUyLTM5ZDAtNDRjYS04OGQ5LWI4ZDZkZGNhMDcwOCIsImV4cCI6IjE3Mjk1OTkzOTAifQ.CgkKBHNuaWQSATgSCwji79-d45q5PRAFGg00NS4xNDYuMjMyLjkyIhRtaWNyb3NvZnQuc2hhcmVwb2ludCosWlRlQ0w4ZXU3WlNPTnVtQjBxcjZjYUp1eERWaVIzWHpaZzZEWkM3dkdNYz0wnQE4AUIQAAAAAAAAAAAAAAAAAAAAAEoQaGFzaGVkcHJvb2Z0b2tlbmIEdHJ1ZWokNjZiNTM3NDctODUyMi00ZWNjLWJmYTctYTk4NWE3YzQyN2I5cikwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDFjOTFiNTI4NkBsaXZlLmNvbXoBMMIBJTAjLmZ8bWVtYmVyc2hpcHx5Y2FvYmRAY29ubmVjdC51c3QuaGvIAQE.P4US2tnR6-_rMV7ULppNIvrDtDxTCt_OYjoZRVOFTEQ"

# 输出文件名
output_file="sunrgbd_v1_revised_0415.tar"

# 使用 curl 下载文件
curl -L -o "$output_file" "$url"

echo "下载完成：$output_file"


# First file
url1="https://hkustconnect-my.sharepoint.com/personal/ycaobd_connect_ust_hk/_layouts/15/download.aspx?UniqueId=2bc2b194-22c6-4489-872d-6591fa2a0906&Translate=false&tempauth=v1.eyJzaXRlaWQiOiJlZjc3ODAyMC0yOTc1LTQ0ZDYtOGFiMC1iYTA1ZjU1ZTViODkiLCJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvaGt1c3Rjb25uZWN0LW15LnNoYXJlcG9pbnQuY29tQDZjMWQ0MTUyLTM5ZDAtNDRjYS04OGQ5LWI4ZDZkZGNhMDcwOCIsImV4cCI6IjE3Mjk2MDAwMDIifQ.CgkKBHNuaWQSATgSCwianvLvkJu5PRAFGg00NS4xNDYuMjMyLjkyIhRtaWNyb3NvZnQuc2hhcmVwb2ludCosaDlVdDZ6blFhc2s2a3I1SEJjZkVvUXNGMkVhWnFUNDJSL1Q1eU4vVldNND0wnQE4AUIQoVxQj6mAADD6dY-w58ITIkoQaGFzaGVkcHJvb2Z0b2tlbmIEdHJ1ZWokNjZiNTM3NDctODUyMi00ZWNjLWJmYTctYTk4NWE3YzQyN2I5cikwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDFjOTFiNTI4NkBsaXZlLmNvbXoBMMIBJTAjLmZ8bWVtYmVyc2hpcHx5Y2FvYmRAY29ubmVjdC51c3QuaGvIAQE.psGkLmAK-jxKZXZqYTRe_Y9LQRQbIpqFgfGsfI_4T1E"
output_file1="scannet200_data.tar.0"
curl -L -o "$output_file1" "$url1"
echo "下载完成：$output_file1"

# Second file
url2="https://hkustconnect-my.sharepoint.com/personal/ycaobd_connect_ust_hk/_layouts/15/download.aspx?UniqueId=11fe7985-e50d-4681-85e6-fada19df058a&Translate=false&tempauth=v1.eyJzaXRlaWQiOiJlZjc3ODAyMC0yOTc1LTQ0ZDYtOGFiMC1iYTA1ZjU1ZTViODkiLCJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvaGt1c3Rjb25uZWN0LW15LnNoYXJlcG9pbnQuY29tQDZjMWQ0MTUyLTM5ZDAtNDRjYS04OGQ5LWI4ZDZkZGNhMDcwOCIsImV4cCI6IjE3Mjk2MDAwOTUifQ.CgkKBHNuaWQSATgSCwjgl_vll5u5PRAFGg00NS4xNDYuMjMyLjkyIhRtaWNyb3NvZnQuc2hhcmVwb2ludCosTk41VGNzcmlsY2gvb2hubkgzU2hpalhZaDNlMVFyVFJjQWozdmVuMTRBZz0wnQE4AUIQAAAAAAAAAAAAAAAAAAAAAEoQaGFzaGVkcHJvb2Z0b2tlbmIEdHJ1ZWokNjZiNTM3NDctODUyMi00ZWNjLWJmYTctYTk4NWE3YzQyN2I5cikwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDFjOTFiNTI4NkBsaXZlLmNvbXoBMMIBJTAjLmZ8bWVtYmVyc2hpcHx5Y2FvYmRAY29ubmVjdC51c3QuaGvIAQE.BG4-RBX5NhJBPJgasTa6A46rmnfsgtv68YRlVQKpor4"
output_file2="scannet200_data.tar.1"
curl -L -o "$output_file2" "$url2"
echo "下载完成：$output_file2"

# Third file
url3="https://hkustconnect-my.sharepoint.com/personal/ycaobd_connect_ust_hk/_layouts/15/download.aspx?UniqueId=5e9160c6-fca7-48a4-ab6c-fc08942a50b5&Translate=false&tempauth=v1.eyJzaXRlaWQiOiJlZjc3ODAyMC0yOTc1LTQ0ZDYtOGFiMC1iYTA1ZjU1ZTViODkiLCJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvaGt1c3Rjb25uZWN0LW15LnNoYXJlcG9pbnQuY29tQDZjMWQ0MTUyLTM5ZDAtNDRjYS04OGQ5LWI4ZDZkZGNhMDcwOCIsImV4cCI6IjE3Mjk2MDAxNDUifQ.CgkKBHNuaWQSATgSCwiqxaDFm5u5PRAFGg00NS4xNDYuMjMyLjkyIhRtaWNyb3NvZnQuc2hhcmVwb2ludCosNTQ3WDZyaXB2NU5LdHNBcDUxR3UxUkVYQXJFclo5SkF6bjVEcm93QUI1dz0wnQE4AUIQAAAAAAAAAAAAAAAAAAAAAEoQaGFzaGVkcHJvb2Z0b2tlbmIEdHJ1ZWokNjZiNTM3NDctODUyMi00ZWNjLWJmYTctYTk4NWE3YzQyN2I5cikwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDFjOTFiNTI4NkBsaXZlLmNvbXoBMMIBJTAjLmZ8bWVtYmVyc2hpcHx5Y2FvYmRAY29ubmVjdC51c3QuaGvIAQE.t4tDCPWgLj9tvRUJbu3aICYHwFKe5rP2scBeqAkLwsc"
output_file3="scannet200_data.tar.2"
curl -L -o "$output_file3" "$url3"
echo "下载完成：$output_file3"
