import torch
from sklearn.metrics import f1_score
import dgl

from utils import load_data, EarlyStopping
import time

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

# def aggregation_computation(tensor_1, tensor_2, tensor_3):
#     t0 = time.time()
#     t1 = time.time()
#     t = t1 - t0

#     return t, tensor
def g_metapath_instance(g, meta_paths):
    pass

def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_data(args['dataset'])

    print("g:",g)
    n_num = g.number_of_edges()
    print("test_num:", n_num)
    meta_paths=[['pa', 'ap'], ['pf', 'fp']]



    # for meta_path in meta_paths:       # 循环每个metapath
    #     print("meta_path:", meta_path)
    #     mp = ['pf']
    #     print("mp:", mp)
    #     t0 = time.time()
    #     new_graph = dgl.metapath_reachable_graph(g, mp) # 得到新图
    #     # 返回edge的数量
    #     new_graph_number_of_edges = new_graph.number_of_edges()
    #     print("new_graph_number_of_edges:", new_graph_number_of_edges)
    #     # 获得paper的nodes
    #     nodelist = new_graph.nodes('paper')
    #     num_nodelist = len(nodelist)
    #     # print("nodelist:",nodelist)
    #     print("num_nodelist:", num_nodelist)
    #     # 循环nodelist中的节点，寻找metapath的下一部分
    #     num_edge_first = 0
    #     num_edge_second = 0
    #     # num_edge_second_rendundancy_free = 0
    #     computation_num = 0
    #     computation_num_redundancy_free = 0
    #     for nl in nodelist:
    #         nl_successors = new_graph.successors(nl)  # nodelist的后继节点
    #         print("nl_successors:", nl_successors)
    #         num_nl_successors = len(nl_successors)    # 后继节点的数量
    #         num_edge_first = num_nl_successors + num_edge_first      # 第一层的edge的数量
    #         print("num_nl_successors:", num_nl_successors)
    #         for nls in nl_successors:
    #             # # 后继节点创建的子图
    #             # nl_subgraph = dgl.node_subgraph(g, nls)
    #             # new_graph_nl_subgraph = dgl.metapath_reachable_graph(nl_subgraph, ['ap'])
    #             # # 获得new_graph_nl_subgraph的edge的数量
    #             # num_edge_nl_subgraph = new_graph_nl_subgraph.number_of_edges()
    #             num_edge_nl_subgraph = len(g.successors(nls, etype='fp'))
    #             print("number_edge_nl_subgraph:", num_edge_nl_subgraph)
    #             num_edge_second = num_edge_second + num_edge_nl_subgraph     # 计算第二层的edge的数量
    #             # num_edge_second_rendundancy_free = num_edge_second - 1 + num_edge_second_rendundancy_free
    #     computation_num = 2 * num_edge_second
    #     computation_num_redundancy_free = num_edge_first + num_edge_second
    #     print("computation_num:", computation_num)
    #     print("ccomputation_num_redundancy_free:", computation_num_redundancy_free)

    #     redundancy_num = computation_num_redundancy_free / computation_num
    #     print("redundancy_num:", redundancy_num)
    #     t1 = time.time()
    #     print("time:", (t1 - t0))
    
    t0 = time.time()
    node_of_paper = g.nodes("paper")
    for np in node_of_paper:
        node_of_author = g.successors(np, etype="pa")
        for na in node_of_author:
            node_of_paper_second = g.successsors(na, etype='ap')
    t1 = time.time()
    print("time:",(t1 - t0))
        








    







    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    if args['hetero']:
        from model_hetero import HAN
        model = HAN(meta_paths=[['pa', 'ap'], ['pf', 'fp']],
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])
        g = dgl.to_homogeneous(g)
        g.ndata['x'] = features 
        print("test:",g)
        print("test finsh...")
        g = g.to(args['device'])

    else:
        from model import HAN
        model = HAN(num_meta_paths=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])
        g = [graph.to(args['device']) for graph in g]




    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))

if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
