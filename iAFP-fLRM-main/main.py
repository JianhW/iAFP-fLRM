
from data_loader import load_bench_data, load_Ind_data
from train import train_test



if __name__ == '__main__':
    train_iter = load_bench_data("Train.csv")
    ind_iter = load_Ind_data('Ind.csv')
    performance, result_bench, roc_data, prc_data = train_test(train_iter, ind_iter)