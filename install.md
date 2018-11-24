## 1. sh install.sh

根据我的经验，使用python3.6,gcc 5.5才成功，因此需要对gcc降级到5.5。


## 2. undefined symbol: THPVariableClass

> `./networks/channelnorm_package/channelnorm.py`

```
ImportError: /.local/lib/python3.6/site-packages/channelnorm_cuda-0.0.0-py3.6-linux-x86_64.egg/channelnorm_cuda.cpython-36m-x86_64-linux-gnu.so: undefined symbol: THPVariableClass
```

[https://blog.csdn.net/xiaomudouer/article/details/84330083](https://blog.csdn.net/xiaomudouer/article/details/84330083)


## 3. 查看光流

> `read`

使用师妹提供的C代码。

[https://blog.csdn.net/qq_20514449/article/details/78907403](https://blog.csdn.net/qq_20514449/article/details/78907403)
