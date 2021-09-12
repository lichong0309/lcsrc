import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
# from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, PPIDataset, GINDataset, TUDataset
# from dgl import DGLGraph

from gcn import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN
import torch.autograd
import torch.nn as nn

from gcn_mp import gcn_msg, gcn_reduce




# def HAggGenerateGraph(g):
#     ### 获得原图g的邻接矩阵
#     g_adj = g.DGLGraph.adjacency_matrix()
#     ### 打印原图g的邻接矩阵
#     print("original graph adjacency matrix:", g_adj)

#     ### 获得HAggregation的邻接矩阵
#     gHA_adjacency_matrix = get_HA_adjacency_matrix(g_adj)

#     ### 构建新图 gHA
    
#     ### 更新原图g


#     return gHA, g



def get_HA_adjacency_matrix(g):
    # ### 获得原图g的邻接矩阵
    # g_adj = g.adjacency_matrix()
    # ### 打印原图g的邻接矩阵
    # print("original graph adjacency matrix:", g_adj)
    # num_adj = len(g_adj[0])
    num_node = g.number_of_nodes()
    Node_same_list = [[],[]]
    for i in range(num_node):
        print("Node {0}-th start....".format(i))
        # temp_i = []     ### 存放i节点非零的节点编号，即节点i的目标节点
        # ### 获得节点i的目标节点
        # for a in range(num_adj):
        #     if g_adj[i][a] == 0:
        #         pass
        #     else:
        #         temp_i.append(a)
        successors_i = g.successors(i)

        for j in range((i+1), num_node):
            # temp_j = []      ### 存放j节点非零的节点编号，即节点j的目标节点
            # ### 获得节点j的目标节点
            # for b in range(num_adj):
            #     if g_adj[j][b] == 0:                
            #         pass
            #     else:
            #         temp_j.append(b)

            successors_j = g.successors(j)

            ### 比较节点i和节点j    
            same_node_i_j = [x for x in successors_i if x in successors_j]
            print("same_Node_i_j:", same_node_i_j)

            ### 节点i和节点j中相同的节点的数量
            ### 如果节点i和节点j相同的节点的数量大于等于2，则符合层次化聚合的要求。
            if len(same_node_i_j) >1:
                Node_same_list[0].append(i)
                Node_same_list[1].append(j)
                ##### 以上步骤获得 Node_same_list, same_node_i_j
                ##### Node_same_list: [[1,3,4],[2,4,6]],表示1和2，3和4，4和6可以层次化聚合

                ### 添加新的节点w，邻居节点：i，j，目标节点same_node_i_j
                g.add_nodes(1)      #### 添加节点
                w = g.number_of_nodes()   ### 添加节点的编号
                g.add_edges([i,j],[w,w])
                src_list = []
                dis_list = same_node_i_j
                for ltemp in range(len(dis_list)):
                    src_list.append(w)
                g.add_edge(src_list, dis_list)
                ### 删除边
                for ltemp in same_node_i_j:
                    edge_id = g.edge_ids(i, ltemp)
                    g.remove_edges(edge_id)
                for ltemp in same_node_i_j:
                    edge_id = g.edge_ids(i, ltemp)
                    g.remove_edges(edge_id)
            else:
                pass
        print("Node i-th finish....")

    return g


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'reddit':
        data = RedditDataset()
    elif args.dataset == 'ppi':
        data = PPIDataset()
    elif args.dataset == 'bzr':
        data = TUDataset(name = 'BZR')
    elif args.dataset == 'imdb':
        data = GINDataset(name = 'IMDBMULTI')
    elif args.dataset == 'collab':
        data = GINDataset(name = 'COLLAB', self_loop = False)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        device = 'cuda:0'
        g = g.int().to(device)

    features = g.ndata['feat']
    labels = g.ndata['label']
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %d
    #   #layer %d
    #   #Train samples %d
    #   #Val samples %d
    #   #Test samples %d""" %
    #       (n_edges, n_classes, args.n_layers,
    #           train_mask.int().sum().item(),
    #           val_mask.int().sum().item(),
    #           test_mask.int().sum().item()))
    print("dataset is :", args.dataset)
    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.to(device)
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)

    if cuda:
        model.to(device)
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)


    node_num = g.number_of_nodes()
    g = get_HA_adjacency_matrix(g)

    # 获得层次化聚合的子图
    node_num_1 = g.number_of_nodes()
    nid_1 = range(node_num, node_num_1)
    g_1 = g.subgraph(nid_1)

    nid_2 = range(node_num)
    g_2 = g.subgraph(nid_2)


    g_1.update_all(gcn_msg, gcn_reduce)
    g_2.update_all(gcn_msg, gcn_reduce)

    features = g.ndata['feat']
    weight = nn.Parameter(torch.Tensor(in_feats, 16))
    g.ndata['h'] = torch.mm(features, weight)
    print("finish .....")




    # # initialize graph
    # dur = []
    # for epoch in range(args.n_epochs):
    #     model.train()
    #     start_time = time.time()
    #     # if epoch >= 3:
    #     #     t0 = time.time()
    #     # forward
    #     logits = model(features)
    #     # loss = loss_fcn(logits[train_mask], labels[train_mask])

    #     # optimizer.zero_grad()
    #     # loss.backward()
    #     # optimizer.step()

    #     # if epoch >= 3:
    #     #     dur.append(time.time() - t0)

    #     # acc = evaluate(model, features, labels, val_mask)
    #     # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
    #     #       "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
    #     #                                      acc, n_edges / np.mean(dur) / 1000))

    # # print()
    # # acc = evaluate(model, features, labels, test_mask)
    # # print("Test accuracy {:.2%}".format(acc))
    #     print("forward finishing...")
    #     fin_time = time.time()
    #     epoch_time = fin_time - start_time
    #     print("epoch time:", epoch_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed', 'coauthor', 'ppi', 'bzr', 'reddit', \
                            'imdb', 'collab').")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=1,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)

















# def main(args):
#     print("main function start...")
#     # load and preprocess dataset
#     if args.dataset == 'cora':
#         data = CoraGraphDataset()
#     elif args.dataset == 'citeseer':
#         data = CiteseerGraphDataset()
#     elif args.dataset == 'pubmed':
#         data = PubmedGraphDataset()
#     else:
#         raise ValueError('Unknown dataset: {}'.format(args.dataset))

#     g = data[0]


#     if args.gpu < 0:
#         cuda = False
#     else:
#         cuda = True
#         print("cuda test...")
#         g = g.int().to(args.gpu)

#     features = g.ndata['feat']
#     labels = g.ndata['label']
#     train_mask = g.ndata['train_mask']
#     val_mask = g.ndata['val_mask']
#     test_mask = g.ndata['test_mask']
#     in_feats = features.shape[1]
#     n_classes = data.num_labels
#     n_edges = data.graph.number_of_edges()
#     print("""----Data statistics------'
#       #Edges %d
#       #Classes %d
#       #Train samples %d
#       #Val samples %d
#       #Test samples %d""" %
#           (n_edges, n_classes,
#               train_mask.int().sum().item(),
#               val_mask.int().sum().item(),
#               test_mask.int().sum().item()))

#     # add self loop
#     if args.self_loop:
#         g = dgl.remove_self_loop(g)
#         g = dgl.add_self_loop(g)
#     n_edges = g.number_of_edges()

#     # normalization
#     degs = g.in_degrees().float()
#     norm = torch.pow(degs, -0.5)
#     norm[torch.isinf(norm)] = 0
#     if cuda:
#         print("cuda test_1....")
#         norm = norm.cuda()
#     g.ndata['norm'] = norm.unsqueeze(1)

#     # create GCN model
#     model = GCN(g,
#                 in_feats,
#                 args.n_hidden,
#                 n_classes,
#                 args.n_layers,
#                 F.relu,
#                 args.dropout)

#     if cuda:
#         model.cuda()
#     loss_fcn = torch.nn.CrossEntropyLoss()

#     # use optimizer
#     optimizer = torch.optim.Adam(model.parameters(),
#                                  lr=args.lr,
#                                  weight_decay=args.weight_decay)

#     # initialize graph
#     dur = []


#     node_num = g.number_of_nodes()
#     g = get_HA_adjacency_matrix(g)

#     # 获得层次化聚合的子图
#     node_num_1 = g.number_of_nodes()
#     nid_1 = range(node_num, node_num_1)
#     g_1 = g.subgraph(nid_1)

#     nid_2 = range(node_num)
#     g_2 = g.subgraph(nid_2)


#     for epoch in range(args.n_epochs):
#         model.train()

#         t0 = time.time()
#         # start = torch.cuda.Event(enable_timing=True)
#         # end = torch.cuda.Event(enable_timing=True)

#         # start.record()
#         # forward
#         # with torch.autograd.profiler.profile(use_cuda = True) as prof:
#         logits = model(features)         
#         # print("test:\n", prof.key_averages().table())
#         loss = loss_fcn(logits[train_mask], labels[train_mask])

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


#         dur.append(time.time() - t0)
#         # end.record()
#         # torch.cuda.synchronize()
#         # dur.append(start.elapsed_time(end))

#         acc = evaluate(model, features, labels, val_mask)
#         print("Epoch {:05d} | Time(ms) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
#               "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
#                                              acc, n_edges / np.mean(dur) / 1000))

#     print()
#     acc = evaluate(model, features, labels, test_mask)
#     print("Test accuracy {:.2%}".format(acc))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='GCN')
#     parser.add_argument("--dataset", type=str, default="cora",
#                         help="Dataset name ('cora', 'citeseer', 'pubmed').")
#     parser.add_argument("--dropout", type=float, default=0.5,
#                         help="dropout probability")
#     parser.add_argument("--gpu", type=int, default=-1,
#                         help="gpu")
#     parser.add_argument("--lr", type=float, default=1e-2,
#                         help="learning rate")
#     parser.add_argument("--n-epochs", type=int, default=1,
#                         help="number of training epochs")
#     parser.add_argument("--n-hidden", type=int, default=16,
#                         help="number of hidden gcn units")
#     parser.add_argument("--n-layers", type=int, default=1,
#                         help="number of hidden gcn layers")
#     parser.add_argument("--weight-decay", type=float, default=5e-4,
#                         help="Weight for L2 loss")
#     parser.add_argument("--self-loop", action='store_true',
#                         help="graph self-loop (default=False)")
#     parser.set_defaults(self_loop=False)
#     args = parser.parse_args()
#     print(args)

#     main(args)
