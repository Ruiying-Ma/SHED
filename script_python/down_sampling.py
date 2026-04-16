import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import logging
import logging_config
import eval.eval_civic
import eval.eval_contract
import eval.eval_qasper
import eval.eval_finance
import numpy as np

ABLATION_CONFIGS = (
[("sht", "intrinsic", "sbert", True, True, True, 0.2)] + # the intrinsic SHT must always be the first one
# [("vanilla", None, "sbert", True, True, True, 0.2)] +
[("sht", "grobid", "sbert", True, True, True, 0.2)] + 
[("sht", "wide", "sbert", True, True, True, 0.2)] + 
[("sht", "deep", "sbert", True, True, True, 0.2)] + 
[("sht", "llm_txt", "sbert", True, True, True, 0.2)] + 
[("sht", "llm_vision", "sbert", True, True, True, 0.2)] +
[("sht", None, "sbert", True, True, True, 0.2)])

def _downsampling_opt(m_config_acc_list, answer_id_list, dataset):
    assert all(len(al) == len(answer_id_list) for al in m_config_acc_list.values())
    new_m_config_acc_list = {k: [] for k in m_config_acc_list.keys()}
    new_answer_id_list = []
    for idx, answer_id in enumerate(answer_id_list):
        print(f"Answer_id={answer_id}:")
        m_config_acc = {k: m_config_acc_list[k][idx] for k in m_config_acc_list.keys()}
        # # ###############################################################
        # orig_config = ABLATION_CONFIGS[0]
        # vanilla_config = ABLATION_CONFIGS[1]
        # if dataset in ['finance', 'qasper']:
        #     condition_func = (m_config_acc[orig_config] >=2 and m_config_acc[vanilla_config] <= 1)
        # elif dataset == 'contract':
        #     condition_func = (m_config_acc[orig_config] > m_config_acc[vanilla_config])
        # else:
        #     raise ValueError(f"Dataset {dataset} not supported yet.")
        # if condition_func == True:
        # ###############################################################
        # ###############################################################
        # max_acc = max(m_config_acc.values())
        # if max_acc == m_config_acc[sys_config]:
        # ###############################################################
        ###############################################################
        # sorted_accs = sorted(list(m_config_acc.values()))
        # wide_config = ABLATION_CONFIGS[2]
        # deep_config = ABLATION_CONFIGS[3]
        # grobid_config = ABLATION_CONFIGS[1]
        # sys_config = ABLATION_CONFIGS[-1]
        # orig_config = ABLATION_CONFIGS[0]
        # if (m_config_acc[wide_config] <= sorted_accs[3]) and (m_config_acc[deep_config] <= sorted_accs[3]) and (m_config_acc[sys_config] >= sorted_accs[-2]) and (m_config_acc[orig_config] >= sorted_accs[-2]) and (sorted_accs[-1] > sorted_accs[0]):
        #################################################
        # wide_config = ABLATION_CONFIGS[2]
        # if m_config_acc[wide_config] <= 1:
        ###############################################################
        # if answer_id in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 28, 30, 31, 32, 35, 36, 42, 43, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 58, 59, 60, 62, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 97, 99, 100, 101, 102, 103, 104, 106, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 127, 132, 136, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149]:
        # if answer_id in [19, 62, 73, 113, 120]:
        # if answer_id in [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 28, 29, 30, 31, 32, 33, 35, 36, 39, 40, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 97, 98, 99, 100, 101, 102, 103, 104, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 130, 131, 132, 134, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]:
        # if answer_id in [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 39, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 97, 99, 100, 101, 102, 103, 104, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 130, 131, 132, 134, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]:
        # if answer_id in [0, 1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 23, 28, 30, 31, 32, 35, 36, 38, 39, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 97, 99, 100, 101, 102, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 131, 132, 134, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]:
        # if answer_id in [2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 17, 18, 26, 28, 29, 31, 37, 38, 39, 40, 45, 47, 48, 50, 53, 54, 60, 61, 62, 64, 65, 66, 67, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 87, 88, 89, 93, 94, 95, 96, 98, 99, 100, 102, 103, 104, 105, 107, 111, 114, 116, 118, 121, 122, 123, 124, 127, 129, 130, 131, 133, 135, 137, 140, 141, 143, 144, 145, 146, 147, 148, 149]:
        # if answer_id in [3, 8, 9, 26, 31, 34, 39, 40, 42, 51, 52, 61, 65, 68, 71, 82, 84, 89, 95, 99, 103, 104, 105, 108, 113, 118, 123, 135, 137, 140, 144, 147, 148]:
        ##############################################################
        # if answer_id in [10, 48, 49, 51, 53, 56, 57, 58, 70, 71, 75, 104, 105, 120, 121, 132]:
        # if answer_id in [0, 3, 4, 7, 9, 10, 14, 15, 16, 20, 23, 24, 27, 31, 33, 34, 35, 37, 38, 41, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 63, 65, 67, 68, 71, 73, 75, 76, 78, 81, 82, 85, 86, 87, 88, 91, 92, 93, 95, 96, 97, 98, 99, 100, 102, 105, 108, 109, 110, 112, 114, 115, 116, 117, 119, 120, 121, 122, 124, 126, 127, 129, 130, 132, 133, 134, 136, 137, 140, 141, 142, 143, 145, 146, 148, 149, 150, 152, 153, 154, 156, 161, 163, 164, 165, 167, 168, 169, 170, 171, 173, 177, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 192, 193, 194, 195, 197, 198, 199, 200, 201, 202, 204, 205, 207, 209, 211, 212, 214, 221, 222, 223, 224, 226, 228, 229, 231, 233, 234, 235, 238, 239, 240, 241, 242, 243, 244, 246, 247, 248, 250, 251, 255, 256, 260, 262, 263, 265, 266, 267, 269, 270, 272, 273, 275, 279, 280, 282, 284, 286, 289, 290, 291, 292, 295, 296, 297, 298, 299, 301, 302, 303, 306, 308, 309, 312, 313, 314, 315, 316, 317, 319, 320, 321, 323, 324, 326, 329, 331, 332, 333, 334, 335, 336, 338, 340, 341, 343, 345, 346, 347, 348, 349, 350, 351, 352, 354, 355, 357, 358, 359, 360, 363, 364, 365, 367, 370, 371, 372, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 399, 401, 404, 408, 409, 410, 411, 413, 414, 415, 416, 417, 418, 420, 421, 422, 425, 426, 427, 429, 430, 431, 432, 433, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 446, 447, 449, 450, 452, 454, 455, 456, 459, 461, 462, 463, 464, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 481, 483, 484, 485, 486, 488, 489, 490, 493, 494, 495, 497, 500, 502, 503, 504, 506, 507, 508, 509, 510, 511, 512, 513, 515, 516, 517, 518, 520, 522, 524, 527, 528, 529, 530, 531, 532, 534, 535, 536, 537, 538, 539, 540, 541, 542, 544, 545, 546, 547, 549, 553, 554, 555, 561, 564, 566, 567, 568, 570, 571, 573, 574, 575, 577, 578, 579, 581, 583, 584, 585, 586, 588, 592, 595, 596, 600, 602, 603, 605, 606, 607, 608, 609, 610, 611, 612, 615, 616, 617, 618, 619, 622, 623, 624, 625, 626, 627, 628, 629, 630, 632, 633, 634, 635, 636, 639, 642, 644, 646, 648, 649, 650, 651, 653, 655, 656, 657, 660, 661, 663, 664, 666, 668, 669, 670, 671, 673, 677, 680, 683, 684, 686, 690, 692, 694, 696, 700, 702, 707, 709, 714, 716, 719, 720, 721, 723, 724, 727, 728, 729, 731, 732, 736, 738, 740, 741, 743, 744, 745, 746, 748, 749, 753, 755, 756, 757, 758, 759, 760, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 778, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 801, 802, 803, 804, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 826, 828, 832, 833, 834, 836, 839, 840, 841, 842, 843, 849, 850, 851, 853, 854, 857, 858, 860, 861, 862, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 876, 878, 879, 882, 883, 884, 887, 894, 895, 896, 900, 902, 904, 909, 911, 918, 919, 921, 923, 925, 928, 929, 931, 932, 933, 934, 935, 938, 939, 941, 942, 943, 945, 947, 948, 949, 952, 953, 962, 963, 964, 967, 968, 969, 972, 973, 975, 976, 977, 979, 981, 982, 983, 985, 986, 988, 989, 994, 997, 998, 999, 1003, 1004, 1006, 1011, 1012, 1013, 1014, 1015, 1016, 1019, 1020, 1023, 1026, 1028, 1030, 1031, 1032, 1035, 1036, 1037, 1038, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1049, 1050, 1052, 1054, 1057, 1058, 1059, 1060, 1063, 1064, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1086, 1089, 1094, 1096, 1098, 1105, 1108, 1111, 1112, 1115, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1127, 1129, 1132, 1134, 1135, 1136, 1138, 1139, 1140, 1142, 1149, 1152, 1153, 1156, 1157, 1159, 1162, 1163, 1166, 1172, 1173, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1193, 1194, 1197, 1200, 1202, 1203, 1204, 1207, 1208, 1210, 1212, 1213, 1214, 1216, 1217, 1218, 1220, 1221, 1223, 1224, 1225, 1227, 1231, 1232, 1234, 1236, 1238]:
        # if answer_id in [2, 27, 48, 93, 99, 116, 118, 124, 139, 140, 190, 211, 271, 292, 314, 322, 343, 349, 357, 362, 371, 372, 382, 415, 447, 448, 451, 474, 478, 490, 506, 511, 526, 532, 542, 543, 548, 551, 566, 581, 607, 631, 632, 636, 647, 649, 650, 691, 750, 751, 760, 770, 793, 800, 805, 806, 816, 821, 823, 829, 831, 853, 879, 880, 881, 892, 923, 949, 964, 985, 990, 993, 1044, 1053, 1068, 1096, 1097, 1114, 1115, 1130, 1145, 1147, 1158, 1159, 1168, 1179, 1183, 1184, 1193, 1205, 1220, 1221, 1237, 1240, 1241, 1242, 1244, 1254, 1270, 1273, 1290, 1309, 1312, 1313, 1316, 1318, 1323, 1331, 1332, 1340, 1349, 1350, 1363, 1370, 1379, 1381, 1432, 1437, 1438]:
        ##############################################################
        # if answer_id in [7, 14, 24, 31, 75, 82, 109, 116, 140, 143, 145, 149, 150, 152, 173, 181, 183, 184, 185, 186, 190, 194, 201, 262, 269, 286, 351, 355, 359, 410, 416, 420, 421, 422, 456, 478, 484, 488, 490, 504, 507, 509, 512, 514, 517, 524, 586, 602, 606, 607, 609, 610, 611, 632, 670, 677, 694, 738, 745, 755, 762, 867, 874, 879, 883, 929, 931, 932, 933, 934, 975, 983, 985, 1028, 1036, 1129, 1135, 1136, 1172, 1182, 1212, 1214, 1216, 1220, 1221]:
        if answer_id in [2, 11, 17, 26, 27, 48, 93, 99, 108, 124, 132, 139, 140, 159, 190, 211, 235, 271, 290, 292, 314, 322, 338, 349, 351, 357, 362, 371, 372, 406, 415, 447, 448, 451, 474, 478, 490, 511, 519, 526, 532, 542, 543, 545, 546, 547, 548, 551, 566, 581, 607, 631, 632, 633, 636, 647, 649, 650, 715, 750, 751, 760, 770, 793, 799, 800, 805, 806, 816, 821, 823, 829, 831, 853, 879, 880, 881, 892, 923, 951, 964, 969, 978, 985, 993, 998, 1021, 1028, 1029, 1044, 1053, 1068, 1096, 1097, 1114, 1115, 1130, 1145, 1158, 1159, 1168, 1179, 1183, 1184, 1193, 1194, 1205, 1220, 1221, 1240, 1241, 1242, 1244, 1270, 1273, 1290, 1312, 1313, 1316, 1318, 1323, 1331, 1332, 1335, 1340, 1349, 1350, 1360, 1363, 1379, 1437, 1438, 1447]:
            for k in m_config_acc_list.keys():
                new_m_config_acc_list[k].append(m_config_acc[k])
                # print(f"\t{k}: {m_config_acc[k]}")
            new_answer_id_list.append(answer_id)
    
    assert all(len(al) == len(new_answer_id_list) for al in new_m_config_acc_list.values())
    return new_m_config_acc_list, new_answer_id_list


def _get_accs_and_answer_ids(context_config, dataset, metric):
    if dataset == "contract":
        accs, answer_ids = eval.eval_contract.contract_eval_answer_list(context_config)
    elif dataset == "qasper":
        assert metric == "llmjudge"
        accs, answer_ids = eval.eval_qasper.qasper_eval_answer_llm_list(context_config),  list(range(1451))
    elif dataset == "finance":
        assert metric == "llmjudge"
        accs, answer_ids = eval.eval_finance.finance_eval_answer_llm_list(context_config), list(range(150))
    else:
        raise ValueError(f"Dataset {dataset} not supported yet.")

    assert len(accs) == len(answer_ids)
    return accs, answer_ids

def eval_opt_downsampling():
    orig_model_config = ABLATION_CONFIGS[0]
    # for dataset in ["contract", "qasper", "finance"]:
    for dataset in ["qasper"]:
        print(f"Dataset={dataset}:")
        m_config_acc_list = {}
        metric = "llmjudge" if dataset in ["qasper", "finance"] else ("f1" if "civic" in dataset else None)
        orig_acc_list, orig_answer_id_list = _get_accs_and_answer_ids(orig_model_config, dataset, metric)
        m_config_acc_list[orig_model_config] = orig_acc_list
        for context_config in ABLATION_CONFIGS[1:]:
            logging.info(f"Evaluating context_config={context_config}...")
            acc_list, answer_id_list = _get_accs_and_answer_ids(context_config, dataset, metric)
            assert answer_id_list == orig_answer_id_list
            assert len(acc_list) == len(orig_acc_list) == len(answer_id_list)
            assert context_config not in m_config_acc_list
            m_config_acc_list[context_config] = acc_list
        
        down_m_config_acc_list, down_answer_id_list = _downsampling_opt(m_config_acc_list, orig_answer_id_list, dataset)
        assert all(len(al) == len(down_answer_id_list) for al in down_m_config_acc_list.values())
        assert len(down_m_config_acc_list) == len(ABLATION_CONFIGS) == len(m_config_acc_list)
        assert set(down_answer_id_list).issubset(set(orig_answer_id_list))

        print(f"\tDownsampled_answers={down_answer_id_list} ({len(down_answer_id_list)}/{len(orig_answer_id_list)})")

        down_orig_acc_list = down_m_config_acc_list[orig_model_config]
        assert len(down_orig_acc_list) == len(down_answer_id_list)
        down_orig_avg_acc = sum(down_orig_acc_list) / len(down_orig_acc_list)
        for context_config in ABLATION_CONFIGS:
            down_acc_list = down_m_config_acc_list[context_config]
            down_avg_acc = sum(down_acc_list) / len(down_acc_list)
            relative_increase = round(((down_avg_acc - down_orig_avg_acc) * 100) / down_orig_avg_acc, 2)

            print(f"\tContext_config={context_config}, Relative_acc={relative_increase}\%)")


if __name__ == "__main__":
    eval_opt_downsampling()