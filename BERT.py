from bert_serving.client import BertClient   #使用bert_as_service调用BERT训练好的词向量
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
corpus = []
bc = BertClient()
km = KMeans(n_clusters=2)
pca = PCA(n_components=2)
gb = open('./data/动物词库.txt',encoding='utf-8').readlines()
for word in gb[:30]:    #为了方便，每个词库只取了前面30个单词
    word = word.split('\t')
    corpus.append(word[0])

fb = open('./data/地名词库.txt',encoding='utf-8').readlines()
for word in fb[:30]:
    word = word.split('\t')
    corpus.append(word[0])

vectors = bc.encode(corpus)
vectors_ = pca.fit_transform(vectors)   #降维到二维
y_ = km.fit_predict(vectors_)       #聚类
plt.rcParams['font.sans-serif'] = ['FangSong']    #支持中文，Linux系统下此方法行不通
plt.scatter(vectors_[:,0],vectors_[:, 1],c=y_)   #将点画在图上
for i in range(len(corpus)):    #给每个点进行标注
    plt.annotate(s=corpus[i], xy=(vectors_[:, 0][i], vectors_[:, 1][i]),
                 xytext=(vectors_[:, 0][i] + 0.1, vectors_[:, 1][i] + 0.1))
plt.show()
