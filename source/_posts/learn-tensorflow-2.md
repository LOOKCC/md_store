---
title: learn_tensorflow-2
date: 2017-12-20 02:33:41
tags: 机器学习
---
# tensorflow学习心得2-各部分联系
上次对tensorflow的总体情况做了综述，这次深入其中，对上次的几个介绍进行深入的介绍，重点介绍Session，Graph，Operator，Tensor之间的联系，以及如何相互调用，形成一个完整的系统。
## Session与Graph
在每个tensorflow的程序里，都会有一张默认的Graph，所有的Graph都必须在Session 中运行，如果没有指定Graph的话，Session会默认为默认的Graph。一个Graph可以在多个Session中运行，但是一个Session只能运行一个Graph。
## Graph与 （Operator和Tensor)
想像一下在数据结构中 学习的图，这里的图中，节点就是Operator,边就是Tensor的流动方向。这里以tf.matmul()举个例子，这是一个计算矩阵乘积的函数，那么，通常情况下，这个Operator就会有两个边指向它，代表要计算的两个矩阵，有一条边出去，代表他计算的结果，这个边的另一端还很有可能指向了其他的节点作为输入。这就代表着Tensor 的流动。总之，这些计算组成了图。
## 总结
总结一下就是Session->Graph->{Tensor，Operator}
这样处理，刚开始对我们来说是比较麻烦的（相比较与Pytorch），同时，因为python运行时检测错误的原因，也给debug带来了困难。但是，瑕不掩瑜，在网络十分复杂的时候，可视化的图是十分清楚的，或者可以这样理解，如果网络非常简单，并且已经写死的话，那么使用Pytorch是一个十分明智的选择，但是，当网络十分复杂，并且需要灵活的改变的时候，Tensorflow明显更适合你。