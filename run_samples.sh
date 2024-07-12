CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset_name sunrgbd_image --if_use_v1 --nqueries 128  --test_ckpt outputs/coda_sunrgbd_stage2/checkpoint_0200.pth --batchsize_per_gpu_test 1 --model_name 3detr_predictedbox_distillation  --ngpus 1 --enc_dim 256 --dec_dim 512  --num_semcls 46 --test_num_semcls 46 --train_range_min 0 --train_range_max 10 --test_range_min 0 --test_range_max 46 --if_input_image --pooling_methods 'average' --cross_enc_nlayers 3  --cross_enc_dim 256 --cross_num_layers 3 --cross_heads 4  --if_clip_more_prompts  --show_only --if_after_nms --show_dir show_outputs/CoDA_sunrgbd  --loss_sem_cls_softmax_weight 1


CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset_name scannet50_image --nqueries 128  --test_ckpt outputs/coda_scannet_stage2/checkpoint_0200.pth --batchsize_per_gpu_test 1  --batchsize_per_gpu 1 --model_name 3detr_predictedbox_distillation  --ngpus 1 --enc_dim 256 --dec_dim 512  --num_semcls 60 --train_range_min 0 --train_range_max 10 --test_range_min 0 --test_range_max 60 --test_num_semcls 60 --if_input_image --pooling_methods 'average' --cross_enc_nlayers 3  --cross_enc_dim 256 --cross_num_layers 3 --cross_heads 4 --if_clip_more_prompts --show_only --if_after_nms --show_dir show_outputs/CoDA_scannet --loss_sem_cls_softmax_weight 1 \
--train_range_list 2 4 5 7 13 15 16 22 56 1163 \
--test_range_list 2 4 5 6 7 8 9 10 11 13 14 15 16 17 18 19 21 22 23 24 26 27 28 29 31 32 33 34 35 36 38  39  40  41  42  44  45  46  47  48  49  50  51  52  54  55  56  57  58  59  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  82  84  86  87  88  89  90  93  95  96  97  98  99  100  101  102  103  104  105  106  107  110  112  115  116  118  120  121  122  125  128  130  131  132  134  136  138  139  140  141  145  148  154  155  156  157  159  161  163  165  166  168  169  170  177  180  185  188  191  193  195  202  208  213  214  221  229  230  232  233  242  250  261  264  276  283  286  300  304  312  323  325  331  342  356  370  392  395  399  408  417  488  540 562  570  572  581  609  748  776  1156  1163  1164  1165  1166  1167  1168  1169  1170  1171  1172  1173  1174  1175  1176  1178  1179  1180  1181  1182  1183  1184  1185  1186  1187  1188  1189  1190  1191 \
--image_size_width 1296 \
--image_size_height 968 \
--reset_scannet_num 50 