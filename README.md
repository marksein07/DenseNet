# [Densely Connected Convolutional Networks (DenseNet)](https://arxiv.org/abs/1608.06993)
Dependency : 
<ol>
    <li>pytorch 1.4.0</li>
    <li>torchvision 0.5.0</li>
    <li>numpy</li>
    <li>tqdm</li>
    <li>[InPalce ABN](https://github.com/mapillary/inplace_abn)</li>
</ol>
<p>
The repositroy training a DenseNet model on Cifar10 Dataset, and could be rewrite easily for ImageNet.

The main goal is training a DenseNet-BC on Cifar10 in single Nivida GPU GTX-1080ti for memory capacity 11 GiB, and the [Inplace ABN](https://arxiv.org/abs/1712.02616) and [Memory-efficient implementation of densenets](https://arxiv.org/abs/1707.06990) are helpful for memory efficient.
</p>

<p>
    To avoid information be transferred through layers by layers, the convolutional layer output is concatenated to all previous layer outputs in the DenseBlock as the inputs to next layer. The generated features channels could be compressed into a very low size. And the connection from each Layer helps to deal with gradient vanishing which is a frequent problem for very Deep Model. 
</p>
<img src="imgs/outline.PNG" alt="outline"/>
<p>
    Affected by ResNet, DenseNet structure could be constructed by DenseBlock, which is divided by pool layers. There are special construction for decreasing FLOPs, Bottlenect layers, and compression rate, both of them have been proved by experiment is beneficial for performance.
</p>
<img src="imgs/DenseNet Structure.PNG" alt="structure"/>
<p>
</p>
<img src="imgs/result.PNG" alt="results table"/>


## Implement Results
<table>
  <tr>
      <td><h2>Loss</h2></td>
      <td><h2>Error rate</h2></td>
  </tr>
  <tr>
      <td><img src="result/loss.png" alt="loss" width="400"/></td>
      <td><img src="result/error_rate.png" alt="error" width="400"/></td>
    </tr>
</table>