import os
import argparse
import platform

DATASETNAME_VS = "M3SVD"

system_name = platform.system()

def parse_args(timestamp_date, timestamp_time):
    log_root = os.path.join("./logs", timestamp_date, timestamp_time)
    os.makedirs(log_root, exist_ok=True)
    print('log_root : {}'.format(log_root))

    parser = argparse.ArgumentParser()
    parser.add_argument('--NEW', type=bool, default=True)
    parser.add_argument('--model', type=str, default='OFVFusion')
    parser.add_argument('--seed', type=int, default=1016)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument('--float16', type=bool, default=False)
    parser.add_argument('--save_comparison_video_num', type=str, default="3")

    if system_name == "Windows":
        print("Windows")
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--workers', type=int, default=0)
        parser.add_argument("--dataset_path_VS", type=str, default=r"F:/DATASETS/IVIF/data/" + DATASETNAME_VS + ".tar")
        parser.add_argument("--test_dataset_path_VS", type=str, default=r"F:\DATASETS\IVIF\test_data") 
        parser.add_argument("--test_Video_path", type=str, default=r"./video_test.tar")
        parser.add_argument("--system_name", type=str, default="Windows")
        parser.add_argument("--indices", type=int, default=10) 
        parser.add_argument("--test_video_path", type=str, default=r"F:/DATASETS/IVIF/video_test/")

    elif system_name == "Linux":
        print("Linux")
        parser.add_argument('--batch_size', type=int, default=4) 
        parser.add_argument('--workers', type=int, default=25)
        parser.add_argument("--dataset_path_VS", type=str, default=r"/root/autodl-fs/" + DATASETNAME_VS + ".tar")
        parser.add_argument("--test_dataset_path_VS", type=str, default=r"/root/autodl-fs/test_data/")
        parser.add_argument("--system_name", type=str, default="Linux")
        parser.add_argument("--indices", type=int, default=1)  
        parser.add_argument('--shutdown', type=bool, default=True)
        parser.add_argument("--test_video_path", type=str, default=r"/root/autodl-fs/video_test/")

    parser.add_argument('--start_epoch', type=int, default=0, metavar='N')
    parser.add_argument("--resume", type=str, default=log_root + "/best_checkpoint.pth", metavar='PATH')
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--test_VS', type=bool, default=False)
    parser.add_argument('--metrics', type=bool, default=True)
    parser.add_argument('--pltshow', type=bool, default=True)
    parser.add_argument("--results_path", type=str, default=r"./results")
    parser.add_argument('--hyperparameters_path', type=str, default=log_root)
    parser.add_argument('--log_dir', type=str, default=log_root, help="Log directory")
    parser.add_argument("--test_save_path", type=str, default=os.path.join(log_root, "results/"))
    parser.add_argument("--video_test_outputs_path", type=str, default=os.path.join(log_root, "video_test_outputs/"))
    parser.add_argument("--epoch_save_path", type=str, default=os.path.join(log_root, "epochs_img/"))
    parser.add_argument("--c", type=str, default="1")

    args = parser.parse_args()

    return args
