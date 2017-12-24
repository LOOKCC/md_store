---
title: learn_tensorflow-4
date: 2017-12-23 23:43:13
tags: 机器学习
---
# tensorflow学习笔记4-cifar10实战
上次是用官网的教程简单的说了一下NMIST的分类，这次将使用tensorflow来完成cifar10 的分来问题。
## 数据集介绍
cifar10 有 60000张32*32的图片，一共有10类，其中50000张作为训练集，10000作为测试集。而这10类分别为[airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck] 更加具体的介绍，可以去它的[官网](https://www.cs.toronto.edu/~kriz/cifar.html)看一下。
## 模型介绍

