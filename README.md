# SynX-NLP

	依赖 jieba, Numpy, SciPy, sklearn, zhon等库。

### 分类示例 
```
>>> import SynXNLP as s
Building prefix dict from the default dictionary ...
Dumping model to file cache C:\Users\Administrator\AppData\Local\Temp\jieba.cache
Loading model cost 0.914 seconds.
Prefix dict has been built succesfully.
Loading vectorizer model from the path E:\repository\SynX-NLP\SynXNLP\feature\vectorizer.model
Loading SVM model from the path E:\repository\SynX-NLP\SynXNLP\classify\SVM.model
>>> s.analysis('test.txt') # 调用analysis方法，进行预测，返回1表示该文本为不良文本，返回-1表示该文本为正常文本
```
---
### 训练示例
	根据 \SynXNLP\data\train\1\ 和 \SynXNLP\data\train\-1\ 中的数据进行训练自动进行训练。
	前者存放的为不良数据集，后者为正常数据集，都应以utf8格式编码。
```
>>> import SynXNLP as s # 自动进行训练
```
