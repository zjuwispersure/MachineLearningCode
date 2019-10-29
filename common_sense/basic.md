



## 损失函数



### Huber Loss

Huber Loss 是一个用于回归问题的带参损失函数, 优点是能增强平方误差损失函数(MSE, mean square error)对离群点的鲁棒性。

当预测偏差小于 δ 时，它采用平方误差,
当预测偏差大于 δ 时，采用的线性误差。

相比于最小二乘的线性回归，HuberLoss降低了对离群点的惩罚程度，所以 HuberLoss 是一种常用的鲁棒的回归损失函数。

Huber Loss 定义如下:
$$
L_\delta(a) = \begin{cases}
\frac{1}{2} a^2, \quad for |a| \le \delta \\
\delta*(|a|-\frac{1}{2}\delta), \quad otherwise
\end{cases}
$$




![huberloss-w450](https://img2018.cnblogs.com/blog/1182370/201809/1182370-20180928094214405-164664611.gif)￼

参数 a 通常表示 residuals，写作 y−f(x)，当 a = y−f(x) 时，Huber loss 定义为：
$$
L_\delta(y,f(x)) = \begin{cases}
\frac{1}{2} (y-f(x))^2, \quad for |y-f(x)| \le \delta \\
\delta*(|y-f(x)|-\frac{1}{2}\delta), \quad otherwise
\end{cases}
$$


δ 是 HuberLoss 的参数，y是真实值，f(x)是模型的预测值, 且由定义可知 Huber Loss 处处可导



### Pseudo-Huber loss 函数

Pseudo-Huber loss 函数可以用作Huber loss 函数的平滑近似，并确保派生物在所有程度上是连续的。 它被定义为
![img](https://images2015.cnblogs.com/blog/1105098/201704/1105098-20170426174309787-1108601851.png)
因此，对于a的小值，该函数近似于a ^ 2 / 2， 对于a的大值该函数近似于具有斜率delta。
虽然上述是最常见的形式，但是还存在Huber损失函数的其他平滑近似





## 损失函数与目标函数 



首先给出结论：损失函数和代价函数是同一个东西，目标函数是一个与他们相关但更广的概念，对于目标函数来说在有约束条件下的最小化就是损失函数（loss function）。

举个例子解释一下:（图片来自Andrew Ng Machine Learning公开课视频）



![img](https://pic3.zhimg.com/50/v2-3f4959cd70308df496ecc4568a0d982d_hd.jpg)![img](https://pic3.zhimg.com/80/v2-3f4959cd70308df496ecc4568a0d982d_hd.jpg)

上面三个图的函数依次为 ![[公式]](https://www.zhihu.com/equation?tex=f_%7B1%7D%28x%29) , ![[公式]](https://www.zhihu.com/equation?tex=f_%7B2%7D%28x%29) , ![[公式]](https://www.zhihu.com/equation?tex=f_%7B3%7D%28x%29) 。我们是想用这三个函数分别来拟合Price，Price的真实值记为 ![[公式]](https://www.zhihu.com/equation?tex=Y) 。

我们给定 ![[公式]](https://www.zhihu.com/equation?tex=x) ，这三个函数都会输出一个 ![[公式]](https://www.zhihu.com/equation?tex=f%28X%29) ,这个输出的 ![[公式]](https://www.zhihu.com/equation?tex=f%28X%29) 与真实值 ![[公式]](https://www.zhihu.com/equation?tex=Y) 可能是相同的，也可能是不同的，为了表示我们拟合的好坏，我们就用一个函数来**度量拟合的程度**，比如：

![[公式]](https://www.zhihu.com/equation?tex=L%28Y%2Cf%28X%29%29+%3D+%28Y-f%28X%29%29%5E2) ，这个函数就称为损失函数(loss function)，或者叫代价函数(cost function)。损失函数**越小**，就代表模型**拟合的越好**。

那是不是我们的目标就只是让loss function越小越好呢？还不是。

这个时候还有一个概念叫风险函数(risk function)。风险函数是损失函数的期望，这是由于我们输入输出的 ![[公式]](https://www.zhihu.com/equation?tex=%28X%2CY%29) 遵循一个联合分布，但是这个联合分布是未知的，所以无法计算。但是我们是有历史数据的，就是我们的训练集， ![[公式]](https://www.zhihu.com/equation?tex=f%28X%29) 关于训练集的**平均损失**称作经验风险(empirical risk)，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7DL%28y_%7Bi%7D%2Cf%28x_%7Bi%7D%29%29) ，所以我们的目标就是最小化 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7DL%28y_%7Bi%7D%2Cf%28x_%7Bi%7D%29%29) ，称为**经验风险最小化**。

到这里完了吗？还没有。

如果到这一步就完了的话，那我们看上面的图，那肯定是最右面的 ![[公式]](https://www.zhihu.com/equation?tex=f_3%28x%29) 的经验风险函数最小了，因为它对历史的数据拟合的最好嘛。但是我们从图上来看 ![[公式]](https://www.zhihu.com/equation?tex=f_3%28x%29)肯定不是最好的，因为它**过度学习**历史数据，导致它在真正预测时效果会很不好，这种情况称为过拟合(over-fitting)。

为什么会造成这种结果？大白话说就是它的函数太复杂了，都有四次方了，这就引出了下面的概念，我们不仅要让经验风险最小化，还要让**结构风险最小化**。这个时候就定义了一个函数 ![[公式]](https://www.zhihu.com/equation?tex=J%28f%29) ，这个函数专门用来度量**模型的复杂度**，在机器学习中也叫正则化(regularization)。常用的有 ![[公式]](https://www.zhihu.com/equation?tex=L_1) , ![[公式]](https://www.zhihu.com/equation?tex=L_2) 范数。

到这一步我们就可以说我们最终的优化函数是：![[公式]](https://www.zhihu.com/equation?tex=min%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7DL%28y_%7Bi%7D%2Cf%28x_%7Bi%7D%29%29%2B%5Clambda+J%28f%29) ，即最优化经验风险和结构风险，而这个函数就被称为**目标函数**。

结合上面的例子来分析：最左面的 ![[公式]](https://www.zhihu.com/equation?tex=f_1%28x%29) 结构风险最小（模型结构最简单），但是经验风险最大（对历史数据拟合的最差）；最右面的 ![[公式]](https://www.zhihu.com/equation?tex=f_3%28x%29) 经验风险最小（对历史数据拟合的最好），但是结构风险最大（模型结构最复杂）;而 ![[公式]](https://www.zhihu.com/equation?tex=f_2%28x%29) 达到了二者的良好**平衡**，最适合用来预测未知数据集。





## 过拟合

#### 过拟合

过拟合是指训练误差和测试误差之间的差距太大。换句换说，就是模型复杂度高于实际问题，**模型在训练集上表现很好，但在测试集上却表现很差**。模型对训练集"死记硬背"（记住了不适用于测试集的训练集性质或特点），没有理解数据背后的规律，**泛化能力差**。

![img](https://pic1.zhimg.com/v2-117601802669511b2e77abcbd78ce414_b.jpg)

**为什么会出现过拟合现象？**

造成原因主要有以下几种：
1、**训练数据集样本单一，样本不足**。如果训练样本只有负样本，然后那生成的模型去预测正样本，这肯定预测不准。所以训练样本要尽可能的全面，覆盖所有的数据类型。
2、**训练数据中噪声干扰过大**。噪声指训练数据中的干扰数据。过多的干扰会导致记录了很多噪声特征，忽略了真实输入和输出之间的关系。
3、**模型过于复杂。**模型太复杂，已经能够“死记硬背”记下了训练数据的信息，但是遇到没有见过的数据的时候不能够变通，泛化能力太差。我们希望模型对不同的模型都有稳定的输出。模型太复杂是过拟合的重要因素。



####  如何防止过拟合？

要想解决过拟合问题，就要显著减少测试误差而不过度增加训练误差，从而提高模型的泛化能力。我们可以使用正则化（Regularization）方法。那什么是正则化呢？**正则化是指修改学习算法，使其降低泛化误差而非训练误差**。

常用的正则化方法根据具体的使用策略不同可分为：（1）直接提供正则化约束的参数正则化方法，如L1/L2正则化；（2）通过工程上的技巧来实现更低泛化误差的方法，如提前终止(Early stopping)和Dropout；（3）不直接提供约束的隐式正则化方法，如数据增强等。

**1. 获取和使用更多的数据（数据集增强）——解决过拟合的根本性方法**

让机器学习或深度学习模型泛化能力更好的办法就是使用更多的数据进行训练。但是，在实践中，我们拥有的数据量是有限的。解决这个问题的一种方法就是**创建“假数据”并添加到训练集中——数据集增强**。通过增加训练集的额外副本来增加训练集的大小，进而改进模型的泛化能力。

我们以图像数据集举例，能够做：旋转图像、缩放图像、随机裁剪、加入随机噪声、平移、镜像等方式来增加数据量。另外补充一句，在物体分类问题里，**CNN在图像识别的过程中有强大的“不变性”规则，即待辨识的物体在图像中的形状、姿势、位置、图像整体明暗度都不会影响分类结果**。我们就可以通过图像平移、翻转、缩放、切割等手段将数据库成倍扩充。

**2. 采用合适的模型（控制模型的复杂度）**

过于复杂的模型会带来过拟合问题。对于模型的设计，目前公认的一个深度学习规律"deeper is better"。国内外各种大牛通过实验和竞赛发现，对于CNN来说，层数越多效果越好，但是也更容易产生过拟合，并且计算所耗费的时间也越长。

根据**奥卡姆剃刀法则**：在同样能够解释已知观测现象的假设中，我们应该挑选“最简单”的那一个。对于模型的设计而言，我们应该**选择简单、合适的模型解决复杂的问题**。

**3. 降低特征的数量**

对于一些特征工程而言，可以降低特征的数量——删除冗余特征，人工选择保留哪些特征。这种方法也可以解决过拟合问题。

**4. L1 / L2 正则化**

**(1) L1 正则化**

在原始的损失函数后面加上一个L1正则化项，即**全部权重** ![[公式]](https://www.zhihu.com/equation?tex=w) **的绝对值的和，再乘以λ/n。**则损失函数变为：

![[公式]](https://www.zhihu.com/equation?tex=C%3DC_%7B0%7D%2B%5Cfrac%7B%5Clambda%7D%7Bn%7D%5Csum_%7Bi%7D%5E%7B%7D%7B%5Cleft%7C+w_%7Bi%7D+%5Cright%7C%7D)

对应的梯度（导数）：

![img](https://pic4.zhimg.com/v2-881d5a0d476c99881ac7d55899aa007b_b.jpg)

 其中 ![[公式]](https://www.zhihu.com/equation?tex=sgn%28w%29) 只是简单地取 ![[公式]](https://www.zhihu.com/equation?tex=w) 各个元素地正负号。

![img](https://pic4.zhimg.com/v2-4a9d62176cb15ed640f1af2c055603fb_b.jpg)

梯度下降时权重 ![[公式]](https://www.zhihu.com/equation?tex=w) 更新变为：

![img](https://pic3.zhimg.com/v2-2b6220b3cba3cbcf04e6c0346df20fc6_b.jpg)

当 ![[公式]](https://www.zhihu.com/equation?tex=w%3D0) 时，|w|是不可导的。所以我们仅仅能依照原始的未经正则化的方法去更新w。

当 ![[公式]](https://www.zhihu.com/equation?tex=w%3E0) 时，sgn( ![[公式]](https://www.zhihu.com/equation?tex=w) )>0, 则梯度下降时更新后的 ![[公式]](https://www.zhihu.com/equation?tex=w) 变小。

当 ![[公式]](https://www.zhihu.com/equation?tex=w%3C0) 时，sgn( ![[公式]](https://www.zhihu.com/equation?tex=w) )>0, 则梯度下降时更新后的 ![[公式]](https://www.zhihu.com/equation?tex=w) 变大。换句换说，**L1正则化使得权重** ![[公式]](https://www.zhihu.com/equation?tex=w) **往0靠，使网络中的权重尽可能为0，也就相当于减小了网络复杂度，防止过拟合。**

这也就是**L1正则化会产生更稀疏（sparse）的解**的原因。此处稀疏性指的是最优值中的一些参数为0。**L1正则化的稀疏性质已经被广泛地应用于特征选择**机制，从可用的特征子集中选择出有意义的特征。

**(2) L2 正则化**

L2正则化通常被称为**权重衰减（weight decay）**，就是在原始的损失函数后面再加上一个L2正则化项，即**全部权重**![[公式]](https://www.zhihu.com/equation?tex=w)**的平方和，再乘以λ/2n**。则损失函数变为：

![img](https://pic4.zhimg.com/v2-03b8b1796c52edf6941d8eba4af3edbb_b.jpg)

对应的梯度（导数）：

![img](https://pic2.zhimg.com/v2-5af458eb3a0ddccc39898ee43187b2ed_b.jpg)

能够发现L2正则化项对偏置 b 的更新没有影响，可是对于权重 ![[公式]](https://www.zhihu.com/equation?tex=w) 的更新有影响：

![img](https://pic3.zhimg.com/v2-10eb4bf30c6929d670b5541616e2cc2a_b.jpg)

这里的![[公式]](https://www.zhihu.com/equation?tex=%5Ceta%E3%80%81n%E3%80%81%5Clambda)都是大于0的， 所以![[公式]](https://www.zhihu.com/equation?tex=1-%5Cfrac%7B%5Ceta%5Clambda%7D%7Bn%7D) 小于1。因此在梯度下降过程中，权重 ![[公式]](https://www.zhihu.com/equation?tex=w) 将逐渐减小，趋向于0但不等于0。这也就是**权重衰减（weight decay）**的由来。

**L2正则化起到使得权重参数 ![[公式]](https://www.zhihu.com/equation?tex=w) 变小的效果，为什么能防止过拟合呢？**因为更小的权重参数 ![[公式]](https://www.zhihu.com/equation?tex=w) 意味着模型的复杂度更低，对训练数据的拟合刚刚好，不会过分拟合训练数据，从而提高模型的泛化能力。

**5. Dropout**

Dropout是在训练网络时用的一种技巧（trike），相当于在隐藏单元增加了噪声。**Dropout 指的是在训练过程中每次按一定的概率（比如50%）随机地“删除”一部分隐藏单元（神经元）。**所谓的“删除”不是真正意义上的删除，其实就是将该部分神经元的激活函数设为0（激活函数的输出为0），让这些神经元不计算而已。

![img](https://pic3.zhimg.com/v2-79650a2b0124214c0bff9fb01e7460ea_b.jpg)

**Dropout为什么有助于防止过拟合呢？**

（a）在训练过程中会产生不同的训练模型，不同的训练模型也会产生不同的的计算结果。随着训练的不断进行，计算结果会在一个范围内波动，但是均值却不会有很大变化，因此可以把最终的训练结果看作是不同模型的平均输出。

（b）它消除或者减弱了神经元节点间的联合，降低了网络对单个神经元的依赖，从而增强了泛化能力。

**6. Early stopping（提前终止）**

对模型进行训练的过程即是对模型的参数进行学习更新的过程，这个参数学习的过程往往会用到一些迭代方法，如梯度下降（Gradient descent）。**Early stopping是一种迭代次数截断的方法来防止过拟合的方法，即在模型对训练数据集迭代收敛之前停止迭代来防止过拟合**。

为了获得性能良好的神经网络，训练过程中可能会经过很多次epoch（遍历整个数据集的次数，一次为一个epoch）。如果epoch数量太少，网络有可能发生欠拟合；如果epoch数量太多，则有可能发生过拟合。Early stopping旨在解决epoch数量需要手动设置的问题。具体做法：每个epoch（或每N个epoch）结束后，在验证集上获取测试结果，随着epoch的增加，如果在验证集上发现测试误差上升，则停止训练，将停止之后的权重作为网络的最终参数。

**为什么能防止过拟合？**当还未在神经网络运行太多迭代过程的时候，w参数接近于0，因为随机初始化w值的时候，它的值是较小的随机值。当你开始迭代过程，w的值会变得越来越大。到后面时，w的值已经变得十分大了。所以early stopping要做的就是在中间点停止迭代过程。我们将会得到一个中等大小的w参数，会得到与L2正则化相似的结果，选择了w参数较小的神经网络。

**Early Stopping缺点：没有采取不同的方式来解决优化损失函数和过拟合这两个问题**，而是用一种方法同时解决两个问题 ，结果就是要考虑的东西变得更复杂。之所以不能独立地处理，因为如果你停止了优化损失函数，你可能会发现损失函数的值不够小，同时你又不希望过拟合。



#### 正则化可以防止过拟合的解释V1

> 过拟合的时候，拟合函数的系数往往非常大，而正则化是通过约束参数的范数使其不要太大，所以可以在一定程度上减少过拟合情况。
>
> 如下图所示，过拟合，就是拟合函数需要顾忌每一个点，最终形成的拟合函数波动很大。在某些很小的区间里，函数值的变化很剧烈。这就意味着函数在某些小区间里的导数值（绝对值）非常大，由于自变量值可大可小，所以只有系数足够大，才能保证导数值很大。

![Alt text](/Users/yanglijuan/Documents/SURE/GITHUB/MachineLearning/common_sense/img/base_img_1.jpg)



#### 正则化防止过拟合全集

评论有个大牛反馈需要看《Understanding machine learning》的第13章《正则化与稳定性(Regularization and Stability)》部分。



线性模型常用来处理回归和分类任务，为了防止模型处于过拟合状态，需要用L1正则化和L2正则化降低模型的复杂度，很多线性回归模型正则化的文章会提到L1是通过稀疏参数（减少参数的数量）来降低复杂度，L2是通过减小参数值的大小来降低复杂度。网上关于L1和L2正则化降低复杂度的解释五花八门，易让人混淆，看完各种版本的解释后过几天又全部忘记了。因此，文章的内容总结了网上各种版本的解释，并加上了自己的理解，希望对大家有所帮助。



- 1、优化角度分析
- 2、梯度角度分析
- 3、先验概率角度分析
- 4、知乎点赞最多的图形角度分析
- 5、限制条件角度分析
- 6、PRML的图形角度分析
- 7、总结

##### 1. 优化角度分析

###### L2正则化的优化角度分析

![img](https://pic3.zhimg.com/v2-ce1e4a1563c517cd77402afd2b64fc16_b.jpg)

在限定的区域，找到使

![img](https://pic3.zhimg.com/v2-d86a685fc1ed5d1da7ba861f750dbc1a_b.jpg)

最小的值。 图形表示为：

![img](https://pic2.zhimg.com/v2-b37fd7f0f78e6917403189d54332f345_b.jpg)



上图所示，红色实线是正则项区域的边界，蓝色实线是

![img](https://pic3.zhimg.com/v2-d86a685fc1ed5d1da7ba861f750dbc1a_b.jpg)

的等高线，越靠里的等高圆，

![img](https://pic3.zhimg.com/v2-d86a685fc1ed5d1da7ba861f750dbc1a_b.jpg)

越小，梯度的反方向是

![img](https://pic3.zhimg.com/v2-d86a685fc1ed5d1da7ba861f750dbc1a_b.jpg)

减小最大的方向，用

![img](https://pic2.zhimg.com/v2-1adf57e7a6061634a873eedf0af03e95_b.jpg)

表示，正则项边界的法向量用实黑色箭头表示。 正则项边界在点P1的切向量有

![img](https://pic3.zhimg.com/v2-d86a685fc1ed5d1da7ba861f750dbc1a_b.jpg)

负梯度方向的分量，所以该点会有往相邻的等高虚线圆运动的趋势；当P1点移动到P2点，正则项边界在点P2的切向量与

![img](https://pic3.zhimg.com/v2-d86a685fc1ed5d1da7ba861f750dbc1a_b.jpg)

梯度方向的向量垂直，即该点没有往负梯度方向运动的趋势；所以P2点是

![img](https://pic3.zhimg.com/v2-d86a685fc1ed5d1da7ba861f750dbc1a_b.jpg)

最小的点。

**结论：L2正则化项使值最小时对应的参数变小。**

######  L1正则化的优化角度分析

![img](https://pic4.zhimg.com/v2-0067c4b0d3ebab60a6f2df6d51dea053_b.jpg)



在限定的区域，找到使

![img](https://pic3.zhimg.com/v2-d86a685fc1ed5d1da7ba861f750dbc1a_b.jpg)

最小的值。

![img](https://pic3.zhimg.com/v2-2f8810e3320ae1b0c34eb2dda42827fe_b.jpg)



**结论：如上图，因为切向量始终指向w2轴，所以L1正则化容易使参数为0，即特征稀疏化。**

##### 2. 梯度角度分析

###### L1正则化

L1正则化的损失函数为：

![img](https://pic3.zhimg.com/v2-e0cb2ff8b59973f804f367e7df78bc5e_b.jpg)



上式可知，当w大于0时，更新的参数w变小；当w小于0时，更新的参数w变大；所以，L1正则化容易使参数变为0，即特征稀疏化。

###### L2正则化

L2正则化的损失函数为：

![img](https://pic2.zhimg.com/v2-6af97f2eb6e793621415a8f5e6f956a9_b.jpg)



由上式可知，正则化的更新参数相比于未含正则项的更新参数多了

![img](https://pic3.zhimg.com/v2-1dca8f57f25b77f372226af91143d336_b.jpg)

项，当w趋向于0时，参数减小的非常缓慢，因此L2正则化使参数减小到很小的范围，但不为0。

##### 3. 先验概率角度分析

文章《深入理解线性回归算法（二）：正则项的详细分析》提到，当先验分布是拉普拉斯分布时，正则化项为L1范数；当先验分布是高斯分布时，正则化项为L2范数。本节通过先验分布来推断L1正则化和L2正则化的性质。 画高斯分布和拉普拉斯分布图（来自知乎某网友）：

![img](https://pic3.zhimg.com/v2-b56f561cb896335c140bdaaff70e656e_b.jpg)



由上图可知，拉普拉斯分布在参数w=0点的概率最高，因此L1正则化相比于L2正则化更容易使参数为0；高斯分布在零附近的概率较大，因此L2正则化相比于L1正则化更容易使参数分布在一个很小的范围内。

##### 4. 知乎点赞最多的图形角度分析

###### 函数极值的判断定理：

（1）当该点导数存在，且该导数等于零时，则该点为极值点； （2）当该点导数不存在，左导数和右导数的符号相异时，则该点为极值点。 如下面两图：

![img](https://pic3.zhimg.com/v2-972439fb9fb5394fbaced4f5c9548162_b.jpg)



上图对应第一种情况的极值，下·图对应第二种情况的极值。本节的思想就是用了第二种极值的思想，只要证明参数w在0附近的左导数和右导数符合相异，等价于参数w在0取得了极值。

###### 图形角度分析

损失函数L如下：

![img](https://pic3.zhimg.com/v2-14e813135ebe1176ab83e3fd1eec28aa_b.jpg)



黑色点为极值点x1，由极值定义：L'(x1)=0；

###### 含L2正则化的损失函数:

![img](https://pic3.zhimg.com/v2-1a4fa6988be4fb42a34cddce444e99c6_b.jpg)

由结论可定性的画含L2正则化的图：

![img](https://pic3.zhimg.com/v2-3a0a6215167fb2b1989cbfd6246a25ae_b.jpg)



极值点为黄色点，即正则化L2模型的参数变小了。

###### 含L1正则化的损失函数:

![img](https://pic2.zhimg.com/v2-ef10904c50f1af283c84563832803d7d_b.jpg)



**因此，只要C满足推论的条件，则损失函数在0点取极值(粉红色曲线），即L1正则化模型参数个数减少了。**

![img](https://pic1.zhimg.com/v2-5d1987d19f171e9afd8c55d60be1a16c_b.jpg)

##### 5. 限制条件法

这种思想还是来自知乎的，觉得很有趣，所以就记录在这篇文章了，思想用到了凸函数的性质。我就直接粘贴这种推导了，若有不懂的地方请私我。

![img](https://pic4.zhimg.com/v2-e8574b24651a5a4273b1dc42e65ee8ff_b.jpg)



**结论**：含L1正则化的损失函数在0点取得极值的条件比相应的L2正则化要宽松的多，所以，L1正则化更容易得到稀疏解（w=0）。

##### 6. PRML的图形角度分析

因为L1正则化在零点附近具有很明显的棱角，L2正则化则在零附近比较平缓。所以L1正则化更容易使参数为零，L2正则化则减小参数值，如下图。

![img](https://pic2.zhimg.com/v2-c6a43abbcebf0d00e4a8d434863e7339_b.jpg)

##### 7. 总结

本文总结了自己在网上看到的各种角度分析L1正则化和L2正则化降低复杂度的问题，希望这篇文章能够给大家平时在检索相关问题时带来一点帮助。若有更好的想法，期待您的精彩回复，文章若有不足之处，欢迎更正指出。

