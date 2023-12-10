CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py --dataset_name sunrgbd_image --nqueries 128  --test_ckpt outputs/coda_sunrgbd_stage1/last_checkpoint.pth  --if_after_nms --model_name 3detr_predictedbox_distillation  --ngpus 8 --enc_dim 256 --dec_dim 512 --train_range_max 10 --test_range_max 46 --num_semcls 46 --test_num_semcls 46 --log_file ./model_release_sunrgbdv1_distillation.lst --if_input_image --pooling_methods 'average' --cross_enc_nlayers 3  --cross_enc_dim 256 --cross_num_layers 3 --cross_heads 4 --cross_enc_nlayers 3 --if_clip_more_prompts  --test_only --batchsize_per_gpu_test 48 --loss_sem_cls_softmax_weight 1 --if_use_v1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py --dataset_name sunrgbd_image --nqueries 128  --test_ckpt outputs/coda_sunrgbd_stage2/checkpoint_0200.pth  --if_after_nms --model_name 3detr_predictedbox_distillation  --ngpus 8 --enc_dim 256 --dec_dim 512 --train_range_max 10 --test_range_max 46 --num_semcls 46 --test_num_semcls 46 --log_file ./model_release_sunrgbdv1_after_3dnod.lst --if_input_image --pooling_methods 'average' --cross_enc_nlayers 3  --cross_enc_dim 256 --cross_num_layers 3 --cross_heads 4 --cross_enc_nlayers 3 --if_clip_more_prompts  --test_only --batchsize_per_gpu_test 48 --loss_sem_cls_softmax_weight 1 --if_use_v1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py --dataset_name sunrgbd_image --nqueries 128  --test_ckpt outputs/baseline_sunrgbd/last_checkpoint.pth  --if_after_nms --model_name 3detrmulticlasshead  --ngpus 8 --enc_dim 256 --dec_dim 512 --train_range_max 10 --test_range_max 46 --num_semcls 46 --test_num_semcls 46 --log_file ./model_release_sunrgbdv1_baseline.lst --if_input_image --pooling_methods 'average' --cross_enc_nlayers 3  --cross_enc_dim 256 --cross_num_layers 3 --cross_heads 4 --cross_enc_nlayers 3 --if_clip_more_prompts  --test_only --batchsize_per_gpu_test 48 --loss_sem_cls_softmax_weight 1 --if_use_v1 --if_with_clip


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 main.py --dataset_name scannet50_image --nqueries 128 \
--test_ckpt outputs/coda_scannet_stage2/checkpoint_0200.pth \
--batchsize_per_gpu_test 48  \
--log_file ./model_release_scannet50_after_3dnod.lst  \
--if_after_nms --model_name 3detr_predictedbox_distillation  --ngpus 8 --enc_dim 256 --dec_dim 512  \
--train_range_min 0 \
--train_range_max 10 \
--test_range_min 0 \
--test_range_max 60 \
--train_range_list 2 4 5 7 13 15 16 22 56 1163 \
--test_range_list 2 4 5 6 7 8 9 10 11 13 14 15 16 17 18 19 21 22 23 24 26 27 28 29 31 32 33 34 35 36 38  39  40  41  42  44  45  46  47  48  49  50  51  52  54  55  56  57  58  59  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  82  84  86  87  88  89  90  93  95  96  97  98  99  100  101  102  103  104  105  106  107  110  112  115  116  118  120  121  122  125  128  130  131  132  134  136  138  139  140  141  145  148  154  155  156  157  159  161  163  165  166  168  169  170  177  180  185  188  191  193  195  202  208  213  214  221  229  230  232  233  242  250  261  264  276  283  286  300  304  312  323  325  331  342  356  370  392  395  399  408  417  488  540 562  570  572  581  609  748  776  1156  1163  1164  1165  1166  1167  1168  1169  1170  1171  1172  1173  1174  1175  1176  1178  1179  1180  1181  1182  1183  1184  1185  1186  1187  1188  1189  1190  1191 \
--if_input_image --pooling_methods 'average' --cross_enc_nlayers 3  --cross_enc_dim 256 --cross_num_layers 3 --cross_heads 4 --cross_enc_nlayers 3 \
--if_clip_more_prompts  --test_only \
--image_size_width 1296 \
--image_size_height 968 \
--dataset_num_workers_test 4 \
--reset_scannet_num 50 \
--num_semcls 60 \
--test_num_semcls 60 \
--loss_sem_cls_softmax_skip_none_gt_sample_weight 1 \
--dist_url tcp://localhost:11116 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 main.py --dataset_name scannet50_image --nqueries 128 \
--test_ckpt outputs/coda_scannet_stage1/last_checkpoint.pth \
--batchsize_per_gpu_test 48  \
--log_file ./model_release_scannet50_before_3dnod.lst \
--if_after_nms --model_name 3detr_predictedbox_distillation  --ngpus 8 --enc_dim 256 --dec_dim 512  \
--train_range_min 0 \
--train_range_max 10 \
--test_range_min 0 \
--test_range_max 60 \
--train_range_list 2 4 5 7 13 15 16 22 56 1163 \
--test_range_list 2 4 5 6 7 8 9 10 11 13 14 15 16 17 18 19 21 22 23 24 26 27 28 29 31 32 33 34 35 36 38  39  40  41  42  44  45  46  47  48  49  50  51  52  54  55  56  57  58  59  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  82  84  86  87  88  89  90  93  95  96  97  98  99  100  101  102  103  104  105  106  107  110  112  115  116  118  120  121  122  125  128  130  131  132  134  136  138  139  140  141  145  148  154  155  156  157  159  161  163  165  166  168  169  170  177  180  185  188  191  193  195  202  208  213  214  221  229  230  232  233  242  250  261  264  276  283  286  300  304  312  323  325  331  342  356  370  392  395  399  408  417  488  540 562  570  572  581  609  748  776  1156  1163  1164  1165  1166  1167  1168  1169  1170  1171  1172  1173  1174  1175  1176  1178  1179  1180  1181  1182  1183  1184  1185  1186  1187  1188  1189  1190  1191 \
--if_input_image --pooling_methods 'average' --cross_enc_nlayers 3  --cross_enc_dim 256 --cross_num_layers 3 --cross_heads 4 --cross_enc_nlayers 3 \
--if_clip_more_prompts  --test_only \
--image_size_width 1296 \
--image_size_height 968 \
--dataset_num_workers_test 4 \
--reset_scannet_num 50 \
--num_semcls 60 \
--test_num_semcls 60 \
--loss_sem_cls_softmax_skip_none_gt_sample_weight 1 \
--dist_url tcp://localhost:11116 \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 main.py --dataset_name scannet50_image --nqueries 128 \
--test_ckpt outputs/baseline_scannet/last_checkpoint.pth \
--batchsize_per_gpu_test 48  \
--log_file model_release_baseline_scannet.lst  \
--if_after_nms --model_name 3detrmulticlasshead  --ngpus 8 --enc_dim 256 --dec_dim 512  \
--train_range_min 0 \
--train_range_max 10 \
--test_range_min 0 \
--test_range_max 60 \
--train_range_list 2 4 5 7 13 15 16 22 56 1163 \
--test_range_list 2 4 5 6 7 8 9 10 11 13 14 15 16 17 18 19 21 22 23 24 26 27 28 29 31 32 33 34 35 36 38  39  40  41  42  44  45  46  47  48  49  50  51  52  54  55  56  57  58  59  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  82  84  86  87  88  89  90  93  95  96  97  98  99  100  101  102  103  104  105  106  107  110  112  115  116  118  120  121  122  125  128  130  131  132  134  136  138  139  140  141  145  148  154  155  156  157  159  161  163  165  166  168  169  170  177  180  185  188  191  193  195  202  208  213  214  221  229  230  232  233  242  250  261  264  276  283  286  300  304  312  323  325  331  342  356  370  392  395  399  408  417  488  540 562  570  572  581  609  748  776  1156  1163  1164  1165  1166  1167  1168  1169  1170  1171  1172  1173  1174  1175  1176  1178  1179  1180  1181  1182  1183  1184  1185  1186  1187  1188  1189  1190  1191 \
--if_input_image --pooling_methods 'average' --cross_enc_nlayers 3  --cross_enc_dim 256 --cross_num_layers 3 --cross_heads 4 --cross_enc_nlayers 3 \
--if_clip_more_prompts  --test_only \
--image_size_width 1296 \
--image_size_height 968 \
--dataset_num_workers_test 4 \
--reset_scannet_num 50 \
--num_semcls 60 \
--test_num_semcls 60 \
--if_with_clip \
--dist_url tcp://localhost:12315 \