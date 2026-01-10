#train_real_list and train_syn_list contain objects having X,A,y
#access using train_real_list[i].X, train_real_list[i].A, train_real_list[i].y
#gnn has a method that reinitialises itself

def distill(train_real_list,train_syn_list,epochs,batch_size,mlp,Q,T,gnn,lr_gnn,lr_X,lr_y,lr_mlp,l_syn,l_q,l_real):
    return train_syn_list, mlp, gnn


