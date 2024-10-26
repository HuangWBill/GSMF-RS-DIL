# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).
import torch
from collections import OrderedDict
import argparse
import warnings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ISPRS')
    return args

def main():
    args = parse_args()
    assert args.dataset_name in ['ISPRS', 'LoveDA'], \
        'AssertionError: Only ["ISPRS", "LoveDA"] supported now.'
    if args.dataset_name == 'ISPRS':
        # ISPRS
        GS_AUFPN_path = r'/result/GS-AUFPN-Potsdam/iter_80000_potsdam.pth'
        GS_AUFPN_unaux_path = r'/result/GS-AUFPN-Potsdam/iter_80000_unaux.pth'
        GSMF_RS_DIL_path = '/result/GSFM-RS-DIL-ISPRS/iter_80000_DIL.pth'

    else:
        # LoveDA
        GS_AUFPN_path = r'/result/GS-AUFPN-LoveDA_Urban/iter_30000_urban.pth'
        GS_AUFPN_unaux_path = r'/result/GS-AUFPN-LoveDA_Urban/iter_30000_unaux.pth'
        GSMF_RS_DIL_path = '/result/GSFM-RS-DIL-LoveDA/iter_30000_DIL.pth'

    checkpoint = torch.load(GS_AUFPN_path)
    state_dict = checkpoint['state_dict']

    keys_to_remove = ['auxiliary_head.conv_seg.weight', 'auxiliary_head.conv_seg.bias',
                      'auxiliary_head.convs.0.conv.weight', 'auxiliary_head.convs.0.bn.weight',
                      'auxiliary_head.convs.0.bn.bias', 'auxiliary_head.convs.0.bn.running_mean',
                      'auxiliary_head.convs.0.bn.running_var', 'auxiliary_head.convs.0.bn.num_batches_tracked']
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]

    torch.save({'state_dict': state_dict}, GS_AUFPN_unaux_path)

    unaux_checkpoint = torch.load(GS_AUFPN_unaux_path)

    state_dict = unaux_checkpoint['state_dict']

    modified_weights = OrderedDict()

    for layer_name, weight in state_dict.items():
        modified_weights[layer_name] = weight

    for layer_name, weight in state_dict.items():
        modified_weights[layer_name.replace('backbone', 'backbone_1').replace('decode_head', 'decode_head_1')] = weight

    torch.save({'state_dict': modified_weights}, GSMF_RS_DIL_path)




if __name__ == '__main__':
    main()
