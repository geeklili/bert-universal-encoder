# bert-universal-encoder
提供了常见bert的通用字符串到编码的预处理方法

#### 项目入口
- 项目的入口文件是：
```
main_fun.py
```

- 项目的主函数只有一个，输入是一个字符串，输出是一段编码，编码的大小可以设置

#### 调用的方法如下
```
pad_size = 32
bert_path = '/opt/app/bert_one_path/'
te = TokenEncode(bert_path, pad_size)
a, b = te.get_encode('我是一只小可爱a')
print(a.shape, b.shape)
```

- bert_path为模型所在的路径
- pad_size为句子的最大长度
