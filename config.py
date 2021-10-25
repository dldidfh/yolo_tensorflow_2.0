import argparse
def parse_args(args):
    parser = argparse.ArgumentParser(description='config info')
    parser.add_argument("--input_scale",default=416, type=int) # 320 + 32*n
    parser.add_argument("--batch_size",default=1, type=int)
    parser.add_argument("--box_per_grid",default=3, type=int)
    parser.add_argument("--class_num",default=7, type=int)  
    parser.add_argument("--epochs",default=10, type=int)   

    parser.add_argument("--train_dir_path",default="dataset/train_data", type=str)
    parser.add_argument("--test_dir_path",default="dataset/test_data", type=str)
    parser.add_argument("--class_name_file",default="dataset/class.names", type=str)

    parser.add_argument("--noobject_weight",default=.5, type=float)   
    parser.add_argument("--coord_weight",default=5. , type=float)

    args = parser.parse_args(args)
    args.grid_size = args.input_scale // 32


    return args