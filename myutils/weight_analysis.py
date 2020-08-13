import torch
# from myutils.utils import data_statics

def load_model_weight(path, weight_name):
    model = torch.load(path)
    return model['state_dict'][weight_name]

def find_term(idx, order=2, origin_dim=784):
    tmp = torch.combinations(torch.arange(origin_dim), order, with_replacement=True)
    return tmp[idx]

if __name__ == '__main__':
    path = '/home/xucong/Log/MNIST/ModeNN/2order/best_98.35.ckpt'
    weight_name = 'fc.weight'
    w = load_model_weight(path, weight_name)
    # data_statics('2order-weight', w, verbose=True)
    feature_dim_norm = torch.norm(w, dim=0)
    # data_statics('feature_dim_norm', feature_dim_norm, verbose=True)
    selected_dim = set()
    test = set()
    x = torch.nonzero(torch.where(feature_dim_norm[784:]>0.4, feature_dim_norm[784:], torch.zeros(feature_dim_norm[784:].shape).cuda())).squeeze()
    idx = torch.combinations(torch.arange(784), 2, with_replacement=True)
    for i in range(len(x)):
        test.add(idx[x[i]][0].item())
        test.add(idx[x[i]][1].item())
    print(test)
    for i in range(784, len(feature_dim_norm)):
        if feature_dim_norm[i] > 0.4:
            tmp = find_term(i - 784, 2)
            selected_dim.add(tmp[0].item())
            selected_dim.add(tmp[1].item())
    print(len(selected_dim))
    print(selected_dim)
    print(test == selected_dim)
    select_mask, _ = torch.sort(torch.Tensor(list(selected_dim)).long())
    print('Done')

    