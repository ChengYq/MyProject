# # coding=utf-8
# # 本模块的作用：模拟grid.py
# # 自己写的好处在于:调整判别标准,去除不需要的模块
# from Algo.libsvmWeight.python.svm import *
# from Algo.libsvmWeight.python.svmutil import *
# from sklearn.metrics import f1_score
# def bestPara(train_path,test_path,W=[]):
#     # 输入的是数据集，标签
#     # 需要是libsvm的输入输出
#     x,y=svm_read_problem(train_path)
#     xt,yt=svm_read_problem(test_path)
#     log2c=range(1,16)
#     log2g=range(-1,-16,-1)
#     for i in log2c:
#         for j in log2g:
#             c=2**i
#             g=2**j
#             prob=svm_problem(W,y,x)
#             p='-c {0} -g {1}'.format(c,g)
#             para=svm_parameter(p)
#             model=svm_train(prob,para)
#             p_label, p_acc, p_val=svm_predict(yt,xt,model)
#             =f1_score(yt,p_label)
#
