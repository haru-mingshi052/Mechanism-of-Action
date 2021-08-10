from sklearn.model_selection import train_test_split

from utils import seed_everything
from model import SwishNN, MishNN
from predict import predict
from train import train_model
from dataprocessing import processing_dl, processing_test_dl, processing_df, make_sub, read_submission, MoADataset_test

#====================
# logの削除
#====================
import warnings
warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser(
    description = "parameter for data processing"
)

parser.add_argument('--data_folder', default='/kaggle/input/lish-moa', type=str,
                    help='データの入っているフォルダ')
parser.add_argument('--output_folder', default='/kaggle/working', type=str,
                    help="提出用ファイルを出力するフォルダ")
parser.add_argument('--hidden_size1', default=200, type=int,
                    help="１つ目のレイヤーサイズ")
parser.add_argument('--hidden_size2', default=150, type=int,
                    help="２つ目のレイヤーサイズ")
parser.add_argument('--activation', default='swish', type=str, choices=['swish', 'mish'],
                    help='活性化関数')
parser.add_argument('--epochs', default=300, type=int,
                    help="学習するエポック数")
parser.add_argument('--es_patience', default=20, type=int,
                    help="何回、改善がないと学習を辞めるか")
args = parser.parse_args()

if __name__ == '__main__':
    seed_everything(71)
    train, test, target = processing_df(args.data_folder)
    tr_x, val_x, tr_y, val_y = train_test_split(train, target, test_size = 0.2, random_state = 71)
    tr_dl, val_dl = processing_dl(tr_x, tr_y, val_x, val_y)

    if args.activation=='swish':
        model = SwishNN(
            n_features = tr_x.shape[1] - 1,
            hidden_size1=args.hidden_size1,
            hidden_size2=args.hidden_size2,
            n_output = tr_y.shape[1] - 1
        )
    elif args.activation=='mish':
        model = MishNN(
            n_features = tr_x.shape[1] - 1,
            hidden_size1=args.hidden_size1,
            hidden_size2=args.hidden_size2,
            n_output = tr_y.shape[1] - 1
        )

    train_model(
        model=model,
        tr_dl=tr_dl,
        val_dl=val_dl,
        output_folder=args.output_folder,
        epochs=args.epochs,
        es_patience=args.es_patience,
    )

    test_ds = MoADataset_test(df=test)

    pred = predict(
        test_ds=test_ds,
        model_path=args.output_folder + 'model_weight.pth'
    )
    
    sub = read_submission(args.data_folder)
    sub = make_sub(
        sub=sub,
        pred=pred
    )
    sub.to_csv(args.output_folder + 'submission.csv', index = False)