from inference import segmentation
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='test', help='Test Path')
    parser.add_argument('--save_path', type=str, default='Label', help='Save Path')

    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_opt()
    test_path = os.path.abspath(opt.test_path)
    save_path = os.path.abspath(test_path + os.path.sep + '..')
    save_path = os.path.join(save_path, 'Label')
    
    segmentation(test_path, save_path)