---
title: learn_tensorflow-1
date: 2017-12-19 07:16:22
tags: 机器学习
---
# tensorflow 学习心得1-框架综述
最近因为一些项目的原因，要学习tensorflow（不过好像早晚都要学）下面就记录一下自己的心得体会。
## 先来说一下tensorflow和pytorch的区别
### 如果要类比的话，pytorch像是C语言，是在按部就班的顺序执行，而tensorflow就像C++一样，有着面向对象的特征。
不同于pytorch类似用到哪里写到那里的方式（基本的pytorch的程序就定义网络，定义Variable的变化，自动求导，基本上就完成了）我们可以从tensorflow中感受到作者对机器学习中的张量的理解，就如同它的名字一样，他希望tensor在计算中像水一样流动。所以tensorflow有了session和graph以及operator的各种属性，来让tensor在其中自由的流动。
## 根据官网解释一些tensorflow名词
### Session  
A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated.  
一个会话对象包裹着一个操作被执行的环境，tensor这其中被评价。  
上面的翻译比较僵硬，但意思就是，这个东西定义了一个上下文，在这个上下文中，有两个东西，一个是节点，或者说是运算符，还有一个就是在这些运算符中流动的tensor，
换话句话说，Session中包含着一个网，在这些网中，由运算构成节点，数据在其中进行流动。所以，我们可以看到，在使用tensorflow的时候，都会创建一个Session
### Graph
A TensorFlow computation, represented as a dataflow graph.  
使用一个数据流图来表示一个tensorflow的计算  
A Graph contains a set of tf.Operation objects, which represent units of computation; and tf.Tensor objects, which represent the units of data that flow between operations.    
一个图包含一系列的用来表示运算单元的operator，和一系列代表着数据并在运算单元之间流动的tendor  
这句话已经说的很明显了，graph就是Session中间包含的网，连接着运算和数据
### Operator
Represents a graph node that performs computation on tensors.  
表示在图像上计算张量的节点。  
An Operation is a node in a TensorFlow Graph that takes zero or more Tensor objects as input, and produces zero or more Tensor objects as output.    
一个运算是一个在tensorflow中的节点，它接受零个或者多个tensor作为输入，输出零个或者多个tensor。  
这个根据上面几个的解释，应该已经很明白了。
### Tensor
Represents one of the outputs of an Operation.  
运算结果之一  
A Tensor is a symbolic handle to one of the outputs of an Operation. It does not hold the values of that operation's output, but instead provides a means of computing those values in a TensorFlow tf.Session.  
张量是操作输出之一的象征性处理，它不是运算输出的结果，而是提供了在TensorFlow tf.Session中计算这些值的方法。
这里我的理解，它可能不是可以直接用的值，但是其中蕴含了返回的值，其实我觉得就可以将它当做值。这样更好理解一些




