# class sklearn.naive_bayes.ComplementNB (alpha=1.0, fit_prior=True, class_prior=None, norm=False)
'''
补集朴素贝叶斯
alpha : 浮点数, 可不填 (默认为1.0)
拉普拉斯或利德斯通平滑的参数 ,如果设置为0则表示完全没有平滑选项。但是需要注意的是,平滑相当于人
为给概率加上一些噪音,因此 设置得越大,多项式朴素贝叶斯的精确性会越低(虽然影响不是非常大),布里
尔分数也会逐渐升高。
norm : 布尔值,可不填,默认False
在计算权重的时候是否适用L2范式来规范权重的大小。默认不进行规范,即不跟从补集朴素贝叶斯算法的全部
内容,如果希望进行规范,请设置为True。
fit_prior : 布尔值, 可不填 (默认为True)
是否学习先验概率
。如果设置为false,则不使用先验概率,而使用统一先验概率(uniform
prior),即认为每个标签类出现的概率是
。
class_prior:形似数组的结构,结构为(n_classes, ),可不填(默认为None)
类的先验概率
。如果没有给出具体的先验概率则自动根据数据来进行计算。

'''
from sklearn.naive_bayes import ComplementNB
from time import time
from sklearn.model_selection import train_test_split
import datetime
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer
# class_1 = 50000 #多数类为50000个样本
# class_2 = 500 #少数类为500个样本
# centers = [[0.0, 0.0], [5.0, 5.0]] #设定两个类别的中心
# clusters_std = [3, 1]
# X, y = make_blobs(n_samples=[class_1, class_2], centers=centers,cluster_std=clusters_std,random_state=0, shuffle=False)
# from sklearn.metrics import brier_score_loss as BS,recall_score,roc_auc_score as AUC
# name = ["Multinomial","Gaussian","Bernoulli","Complement"]
# models = [MultinomialNB(),GaussianNB(),BernoulliNB(),ComplementNB()]
# for name,clf in zip(name,models):
#     times = time()
#     Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3, random_state=420)
# #预处理
#     if name != "Gaussian":
#         kbs = KBinsDiscretizer(n_bins=10, encode='onehot').fit(Xtrain)
#         Xtrain = kbs.transform(Xtrain)
#         Xtest = kbs.transform(Xtest)
#     clf.fit(Xtrain, Ytrain)
#     y_pred = clf.predict(Xtest)
#     proba = clf.predict_proba(Xtest)[:, 1]
#     score = clf.score(Xtest, Ytest)
#     print(name)
#     print("\tBrier:{:.3f}".format(BS(Ytest, proba, pos_label=1)))
#     print("\tAccuracy:{:.3f}".format(score))
#     print("\tRecall:{:.3f}".format(recall_score(Ytest, y_pred)))
#     print("\tAUC:{:.3f}".format(AUC(Ytest, proba)))
#     print(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f"))

'''
    Multinomial
	Brier:0.007
	Accuracy:0.990
	Recall:0.000
	AUC:0.991
00:00:090469
Gaussian
	Brier:0.006
	Accuracy:0.990
	Recall:0.438
	AUC:0.993
00:00:021272
Bernoulli
	Brier:0.009
	Accuracy:0.987
	Recall:0.771
	AUC:0.987
00:00:038716
Complement
	Brier:0.038
	Accuracy:0.953
	Recall:0.987
	AUC:0.991
00:00:035113
'''
# 贝叶斯做文本分类
sample = ["Machine learning is fascinating, it is wonderful", "Machine learning is a sensational techonology", "Elsa is a popular character"]
# from sklearn.feature_extraction.text import CountVectorizer
# vec = CountVectorizer()
# X = vec.fit_transform(sample)
# print(X)
import pandas as pd
# 注意稀疏矩阵是无法输入pandas的
# CVresult = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
# print(CVresult)
'''
TF-IDF
TF-IDF全称term frequency-inverse document frequency,词频逆文档频率,是通过单词在文档中出现的频率来衡
量其权重,也就是说,IDF的大小与一个词的常见程度成反比,这个词越常见,编码后为它设置的权重会倾向于越
小,以此来压制频繁出现的一些无意义的词。在sklearn当中,我们使用feature_extraction.text中类TfidfVectorizer
来执行这种编码。
'''
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
# vec = TFIDF()
# X = vec.fit_transform(sample)
# # print(X)
# # 同样使用接口get_feature_names()调用每个列的名称
# TFIDFresult = pd.DataFrame(X.toarray(),columns=vec.get_feature_names())
# print(TFIDFresult)
# # 使用TF-IDF编码之后,出现得多的单词的权重被降低了么?
# print(TFIDFresult.sum(axis=0) / TFIDFresult.sum(axis=0).sum())
from sklearn.datasets import fetch_20newsgroups
#初次使用这个数据集的时候,会在实例化的时候开始下载
# data = fetch_20newsgroup()
import numpy as np
import pandas as pd
# 现在我们就可以直接通过参数来提取我们希望得到的数据集了。
# 对于假设样本之间互相独立并且服从相同分布的算法或模型(比如随机梯度下降)来说可能很重要。
categories = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "talk.politics.mideast"]
train = fetch_20newsgroups(subset="train", categories=categories)
test = fetch_20newsgroups(subset="test", categories=categories)
# print(train)
# print("*************")
# 随意提取一篇文章来看看
# print(train.data[0])
# print("*************")
# 查看一下我们的标签
# print(np.unique(train.target))
# print("*************")
# print(len(train.target))
# print("*************")
# 是否存在样本不平衡问题?
# for i in [0, 1, 2, 3]:
#     print(i, (train.target == i).sum()/len(train.target))
'''
0 0.26052974381241856
1 0.25749023013460703
2 0.23708206686930092
3 0.24489795918367346
'''
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
Xtrain = train.data
Xtest = test.data
Ytrain = train.target
Ytest = test.target
tfidf = TFIDF().fit(Xtrain)
Xtrain_ = tfidf.transform(Xtrain)
Xtest_ = tfidf.transform(Xtest)
# print(Xtrain_)
tosee = pd.DataFrame(Xtrain_.toarray(), columns=tfidf.get_feature_names())
# print(tosee.shape) # (2303, 40725)
# print(tosee.head())

# 建模
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import brier_score_loss as BS
name = ["Multinomial","Complement", "Bournulli"] #
# 注意高斯朴素贝叶斯不接受稀疏矩阵
models = [MultinomialNB(), ComplementNB(), BernoulliNB()] #
for name, clf in zip(name, models):
    clf.fit(Xtrain_, Ytrain)
    y_pred = clf.predict(Xtest_)
    proba = clf.predict_proba(Xtest_)
    score = clf.score(Xtest_, Ytest)
    print(name)
    print(y_pred)
    print('#######################')
    # print(proba)
    # print('#######################')
    print(score)
    print('#######################')
# 4个不同的标签取值下的布里尔分数
    Bscore = []
    for i in range(len(np.unique(Ytrain))):  # 0,1,2,3
        bs = BS(Ytest, proba[:, i], pos_label=i)
        Bscore.append(bs)
        print("\tBrier under {}:{:.3f}".format(train.target_names[i], bs))
    print("\tAverage Brier:{:.3f}".format(np.mean(Bscore)))
    print("\tAccuracy:{:.3f}".format(score))
    print("\n")


