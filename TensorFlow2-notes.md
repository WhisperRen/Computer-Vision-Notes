# TensorFlow 2-Notes

## TF2  基础

* numpy() 方法：对于一个 Tensor，具有 `numpy()` 方法，可以将 Tensor 的值转换为一个 ndarray

* tf.GradientTape()：eager 模式下用来记录梯度的类。只要进入了`with tf.GradientTape() as tape`的上下文环境，则该环境中的计算步骤都会被自动记录下来。离开上下文后停止记录，但记录器仍然可用，因此可以通过`grads = tape.gradient(loss, model.trainable_variables)`来求损失函数对模型可训练变量的梯度

* `optimizer = tf.keras.optimizers.SGD()`，实例化一个优化器，通过

  `optimizer.apply_gradients(zip(grads, model.trainable_variables))`依据梯度更新模型的可训练参数

* Python 的 argparse 模块：在 TF2 中去掉了 flags 模块，转为使用 Python 自己的命令行参数模块：

  ```python
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', default='train', help='train or test')
  parser.add_argument('--batch_size', default=32)
  parser.add_argument('--learning_rate', default=0.0001)
  args = parser.parse_args()
  ```

  

## TF2 模型建立与训练

* 两个重要概念：**Model and Layer**

* 模型以类的形式呈现，通过继承 `tf.keras.Model` 来自定义模型，在子类中需要重写 `__init__`和 `call(input)`两个方法，基本格式：

  ```python
class MyModel(tf.keras.Model):
      def __init__(self):
          super().__init__()
          self.layer1 = tf.keras.layers.Conv2D()
          '''and other layers, paras defination in the model'''
      
      def call(self, input):
          x = self.layer1(input)
          return tf.nn.softmax(x)
  ```
  
* 有两个交叉熵损失函数类，`tf.keras.losses.SparseCategoricalCrossentropy()`，和 `CategoricalCrossentropy()`，**sparse 的含义是真是标签 y_true 可以传入 int 类型的标签**

  ```python
with tf.GradientTape() as tape:
      predictions = model(images)
      loss = loss_obj(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.tranable_variables))
  ```
  
* 自定义层：自定义层需要继承 `tf.keras.layers.Layer` 类，并重写`__init__`、`build` 和 `call`三个方法

  ```python
  class MyLayer(tf.keras.layers.layer):
      def __init__(self):
          super().__init__()
          '''and so on'''
      
      def build(self, input_shape):
          # input_shape 是第一次运行call()时参数inputs的shape，无需显式定义
          self.variable0 = self.add_weight('v0')
          self.variable1 = self.add_weight('v1')
          
      def call(self, inputs):
          return outputs
  ```

* 自定义损失函数需要继承 `tf.keras.losses.Loss` 类，重写 `call`方法即可，如下定义均方差损失：

  ```python
  class MeanSquaredError(tf.keras.losses.Loss):
      def call(self, y_true, y_pred):
          return tf.reduce_mean(tf.square(y_pred - y_true))
  ```

* 自定义评估指标需要继承 `tf.keras.metrics.Metric` 类，并重写`__init__`、`update_state`和`result`三个方法，下面对 SparseCategoricalAccuracy 做了简单的重实现：

  ```python
class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
      def __init(self):
          super().__init__()
          self.init = tf.zeros_initializer()
          self.total = self.add_weight(name='total', dtype=tf.int32,
                                       initializer=self.init)
          self.count = self.add_weight(name='count', dtype=tf.int32,
                                       initializer=self.init)
      
      def update_state(self, y_true, y_pred, sample_weight=None):
          pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
          values = tf.cast(tf.equal(y_true, pred), tf.int32)
          self.total.assign_add(tf.shape(y_true)[0])
          self.count.assign_add(tf.reduce_sum(values))
      
      def result(self):
          return self.count / self.total
  ```
  
* 使用 keras 中预定义的卷积网络结构

  预定义的经典卷积网络结构，如 VGG16 等封装在`tf.keras.applications`中，例如：

  `model = tf.keras.applications.MobileNetV2()`

  每个网络有自己特定的参数，以下是一些共通的常用参数：

  1. `input_shape`：大多数默认为 224x224x3，一般模型对输入的大小有下限，至少 32x32 或 75x 75
  2. `include_top`：在网络的最后是否包含全连接层，默认为 True
  3. `weights`：预训练权值，默认为 'imagenet'，如需随机初始化变量可设为 None
  4. `classes`：默认为 1000，如需修改该参数需`include_top=True`，`weights=None`

## TF2 常用模块

### tf.train.Checkpoint

* TF 提供了`tf.train.Checkpoint`类来保存和恢复模型变量，使用其`save`和`restore`方法可以将 TF 中所有包含 **CheckPointable State** 的对象进行保存和恢复。使用时，首先实例化一个Checkpoint

  `checkpoint = tf.train.Checkpoint(model=model)` 

  tf.train.Checkpoint() 接受的初始化参数是一个 \**kwargs，具体而言，是一系列的**键值对，键名可以随意取，值为需要保存的对象**。例如，假设想要保存一个继承`tf.keras.Mode`的模型 model 和一个继承`tf.train.Optimizer`的优化器 optimizer，可以这样写：

  `checkpoint = tf.train.Checkpoint(myModel=model, myOptimizer=optimizer)`

  **注意，在恢复变量时将要使用这些键名**。当模型训练完成需要保存时，使用：

  `checkpoint.save(save_path_with_prefix)`

  save_path_with_prefix 是保存文件的目录+前缀。例如在源码目录下建立一个名为 save 的文件夹并调用一次 `checkpoint.save('./save/model/ckpt')`，就可以在 save 目录下找到名为 checkpoint、**model.ckpt-1.index** 和 model.ckpt-1.data-00000-of-00001 的三个文件。`checkpoint.save()`**方法可以运行多次，每运行一次都会得到一个 .index 文件和 .data 文件，序号依次累加**

  当要恢复模型参数时，再次实例化一个 Checkpoint，**同时保持键名一致**，再调用 `restore()`方法：

  ```python
model_to_be_restored = myModel()
  checkpoint = tf.train.Checkpoint(myModel=model_to_be_restored)
checkpoint.restore(save_path_with_prefix_and_index)
  ```

  例如，调用 `checkpoint.restore('./save/model.ckpt-1')`就可以载入名称为 model.ckpt，序号为 1 的文件来恢复模型

* 当想恢复许多检查点中最近的一个时，可以使用 `tf.train.latest_checkpoint(save_path)`，它返回最近一个检查点的名字：

  ```python
model = myModel()
  checkpoint = tf.train.Checkpoint(myModel=model)
  checkpoint.restore(tf.train.latest_checkpoint('./save'))
  ```
  
* 使用 `tf.train.CheckpointManager`删除旧的 Checkpoint 及自定义检查点编号

  在实例化一个 Checkpoint 之后，接着定义一个 CheckpointManager：

  ```python
  checkpoint = tf.train.Checkpoint(model=model)
  manager = tf.train.CheckpointManager(checkpoint, directory='./save',
                                       checkpoint_name='model.ckpt',
                                       max_to_keep=k)
  ```

  当要保存模型时，直接调用 `manager.save()`即可，若想要自定义保存的 Checkpoint 的编号，可以指定 **checkpoint_number** 参数，如：`manager.save(checkpoint_number=batch_index)`

### TensorBoard：训练过程可视化

#### 实时查看参数变化情况

* 首先实例化一个记录器，并指定 TensorBoard 记录文件的存放目录，接下来当需要记录训练过程中的参数时，通过 `with`语句指定希望使用的记录器类型，并对需要记录的参数（一般为 scalar）运行：

  `tf.summary.scalar(name, tensor, step=batch_index)`

  即可将训练过程中参数在 step 时刻的值记录下来，一般将 step 设置为训练过程中的 batch 序号

  ```python
  summary_writer = tf.summary.create_file_writer('./tensorboard_files')
  
  for batch_index in range(num_batches):
      with summary_writer.as_default():                      # 指定希望使用的记录器
          tf.summary.scalar('loss', loss, step=batch_index)
          tf.summary.scalar('MyScalar', my_scalar, step=batch_index)
  ```

  每运行一次 `tf.summary.scalar()`，记录器就会向记录文件中写入一条记录。除了 scalar 外，也可以对其它数据类型，如图像等进行可视化

* 当要对训练过程可视化时，在代码目录打开终端，运行

  `tensorboard --logdir=./tensorboard_files`

  即可以用浏览器可视化保存的数据。默认情况下，TensorBoard 每 30 秒更新一次数据，但也可以点击刷新按钮手动刷新

* **注意事项：**

  1. 如果需要重新训练，需要删除掉记录文件夹中的信息并重启 TensorBoard（或者建立一个新的记录文件）
  2. 记录文件完整路径保持全英文

#### 查看 Graph 和 Profile 信息

* 还可以在训练开始前通过使用 `tf.summary.trace_on`来开启 Trace，之后 TensorFlow 会记录下训练时的大量信息（如计算图结构、每个 operation 的耗时等）。在训练完成后，使用 `tf.summary.trace_export`将记录结果输出至文件

  ```python
  tf.summary.trace_on(graph=True, profiler=True)  # 开启trace，可以选择是否要记录图结构和profile信息
  '''
  进行训练
  '''
  with summary_writer.as_default():
      tf.summary.trace_export(name='model_trace', step=0, profiler_outdir=log_dir)
  ```

### tf.data：数据集的创建与预处理

#### 数据集对象的建立

* `tf.data`的核心是 `tf.data.Dataset`类，提供了对数据集的高层封装，其由一系列可迭代访问的 element 组成，每个元素包含一个或多个张量。比如一个由图片组成的数据集，每个元素可以是一个形状为 H x W x C 的图片张量，也可以是由图片张量和对应标签张量组成的 tuple

* `tf.data.Dataset.from_tensor_slices()`：该方法是建立 Dataset 最基础的方法，适用于 **数据量较小（能够整个装进内存）的情况**。

  具体而言，如果数据集中的所有元素是通过张量的第 0 维拼接起来的，那么提供一个这样的张量或者几个第 0 维相同的张量作为输入，即可以按所有张量的第 0 维展开来构建 Dataset，数据集的元素数量即为张量第 0 维的大小（例如 MNIST 训练集即为一个 [60000, 28, 28, 1] 的张量，表示了 60000 张 28x28x1 的灰度图像）

* **注意：当提供多个张量作为输入时，它们的第 0 维必须相同，且必须将它们作为元组拼接作为输入**

  ```python
  (train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
  # [60000,28,28]->[60000,28,28,1]
  train_data = np.expand_dims(train_data.astype(np.float32)/255.0, axis=-1)
  mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
  ```

#### 数据集对象的预处理

`tf.data.Dataset`类提供了多种数据集预处理方法，常用的如：

* `Dataset.map(f)`：对数据集中的每个元素应用函数 f，从而得到一个新的数据集（这部分往往结合 `tf.io`进行读写和解码文件，`tf.image`进行图像处理）
* `Dataset.shuffle(buffer_size)`：将数据集打乱（设定一个固定大小的缓冲区（buffer），取出前 buffer_size 个元素放入，并从缓冲区中随机采样，采样后的数据用后续数据替换，也就是 buffer_size 越大，越随机）
* `Dataset.batch(batch_size)`：将数据分批，即对每 batch_size 个元素使用 tf.stack() 在第 0 维合并，成为一个新的元素

以 MNIST 数据集为例：

* 使用 `Dataset.map()`将所有图片旋转 90 度：

```python
def rot90(image, label):
    image = tf.image.rot90(image)
    return image, label

mnist_dataset = mnist_dataset.map(rot90)
```

* 使用 `Dataset.batch()`将数据集划分批次，每个 batch 大小为 4：

```python
mnist_dataset = mnist_dataset.batch(4)
for _ in range(num_epoch):
    for images, labels in mnist_dataset:
        train_step(images, labels)
```

#### tf.data 的并行化

* `Dataset.prefetch()`：该方法使得我们可以让数据集对象 Dataset 在训练时预取出若干个元素，使得在 GPU 训练的同时 CPU 可以准备数据，从而提升训练流程的效率

  它的使用方法与 `Dataset.batch()`等非常类似，若希望开启预加载数据，使用以下代码即可：

  ```python
  mnist_dataset = mnist_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  ```

  其中参数 buffer_size 既可以手动设置，也可以设置为 `tf.data.experimental.AUTOTUNE`从而由 TensorFlow 自动选取合适的值

* `Dataset.map(num_parallel_calls)`：Dataset.map() 也可以利用多 CPU 资源并行地对数据进行变换，只需指定 num_parallel_calls 参数即可。假设用于训练的 CPU 具有四颗核心，希望利用多核心对图像进行旋转 90 度的操作，代码如下：

  ```python
  mnist_dataset = mnist_dataset.map(map_func=rot90, num_parallel_calls=4)
  ```

  这里也可以将 num_parallel_calls 指定为 `tf.data.experimental.AUTOTUNE`来让 TensorFlow 选择合适的值

#### Dataset 元素的获取与使用

`tf.data.Dataset`类对象本身是一个 Python 的 **可迭代对象**，因此可以使用 For 循环迭代：`for a, b, c in dataset`

* `iter()`：也可以使用 `iter()`显示地创建一个 Python 迭代器并使用 `next()`方法获取下一元素：

  ```python
  dataset = tf.data.Dataset.from_tensor_slices((A, B, C))
  it = iter(dataset)
  a0, b0, c0 = next(it)
  a1, b1, c1 = next(it)
  ```

#### cats_vs_dogs 图像分类实例

这里把原书中的代码贴在这里：

```python
import tensorflow as tf
import os
num_epochs = 10
batch_size = 32
learning_rate = 0.001
data_dir = 'C:/datasets/cats_vs_dogs'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
test_cats_dir = data_dir + '/valid/cats/'
test_dogs_dir = data_dir + '/valid/dogs/'

def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
    return image_resized, label

if __name__ == '__main__':
    train_cat_filenames = tf.constant([train_cats_dir + name for name in os.listdir(train_cats_dir)])
    train_dog_filenames = tf.constant([train_dogs_dir + name for name in os.listdir(train_dogs_dir)])
    train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis=-1)
    # 将标签二值化
    train_labels = tf.concat([tf.zeros(train_cat_filenames.shape, dtype=tf.int32),
                             tf.ones(train_dog_filenames.shape, dtype=tf.int32)], axis=-1)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    # 并行地将构建Dataset
    train_dataset = train_dataset.map(map_func=_decode_and_resize,
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(20000).batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    model = tf.keras.Sequential()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                 loss=tf.keras.losses.sparse_categorical_crossentropy)
    
    model.fit(train_dataset, epochs=num_epochs)
```

### TFRecord 的简单使用

TFRecord 可以理解为一系列 **序列化** 的 `tf.train.Example`元素所组成的列表文件，而每一个 `tf.train.Example`又由若干个 `tf.train.Feature`的字典组成。形式如下：

```python
# dataset.tfrecords
[
    {
        # example 1 (tf.train.Example)
        'feature_1': tf.train.Feature,
        'feature_k': tf.train.Feature
    },
    
    {
        # example N (tf.train.Example)
        'feature_1': tf.train.Feature,
        'feature_k': tf.train.Feature
    }
]
```

为了将形式各样的数据集整理为 TFRecord 格式，可以对数据集中的每个元素进行以下步骤：

1. 读取该数据元素到内存
2. 将该元素转换为 `tf.train.Example`对象（每一个 `tf.train.Example`由若干个 `tf.train.Feature`的字典组成，因此需要先建立 Feature 的字典
3. 将该 `tf.train.Example`对象序列化为字符串，并通过一个预先定义的 `tf.io.TFRecordWriter`写入 TFRecord 文件

而读取 TFRecord 文件数据则可按照以下步骤：

1. 通过 `tf.data.TFRecordDataset`读入原始的 TFRecord 文件（此时文件中的 `tf.train.Example`对象尚未被反序列化），获得一个 `tf.data.Dataset`对象
2. 通过 `Dataset.map()`方法，对该数据集对象中的每一个序列化的 `tf.train.Example`字符串执行 `tf.io.parse_single_example`函数，从而实现反序列化

#### 将数据集存储为 TFRecord 文件

以下以 cats_vs_dogs 二分类数据集的训练集为例，展示如何将该数据集转换为 TFRecord 文件

首先初始化数据集的图片文件名列表和标签：

```python
import tensorflow as tf
import os

data_dir = 'C:/datasets/cats_vs_dogs'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
tfrecord_name = data_dir + '/train/train.tfrecords'

train_cat_filenames = [train_cats_dir + filename for filename in os.listdir(train_cats_dir)]
train_dog_filenames = [train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)]
train_filenames = train_cat_filenames + train_dog_filenames
train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)
```

然后迭代读取图片，建立 `tf.train,Feature`字典和 `tf.train.Example`对象，序列化并写入 TFRecord 文件：

```python
with tf.io.TFRecordWriter(tfrecord_name) as writer:
    for filename, label in zip(train_filenames, train_labels):
        # 以Byte格式读取image
        image = open(filename, 'rb').read()
        # 建立tf.train.Feature字典
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }
        # 通过字典建立Example
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # 将Example序列化并写入TFRecord文件
        writer.write(example.SerializeToString())
```

这里，`tf.train.Feature`支持三种数据格式：

1. `tf.train.BytesList`：字符串或原始 Byte 文件（如图片），通过 bytes_list 参数传入一个由字符串数组初始化的 `tf.train.BytesList`对象
2. `tf.train.FloatList`：浮点数，通过 float_list 参数传入一个由浮点数数组初始化的 `tf.train.FloatList`对象
3. `tf.train.Int64List`：整数，通过 int64_list 参数传入一个由整数数组初始化的 `tf.train.Int64List`对象

#### 读取 TFRecord 文件

以下代码可以读取之前建立的 TFRecord 文件，并通过 `Dataset.map`方法，使用 `tf.io.parse_single_example`函数对数据集中序列化的 `tf.train.Example`对象解码：

```python
# 读取TFRecord文件为一个Dataset
raw_dataset = tf.data.TFRecordDataset(tfrecord_name)

# 定义Feature结构，告知解码器每个Feature的类型
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

# 将TFRecord中序列化的Example解码
def _parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    # 解码JPEG图片
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
    return feature_dict['image'], feature_dict['label']

dataset = raw_dataset.map(_parse_example)
```

这里的 feature_description 类似于一个数据集的描述文件，通过一个由键值对组成的字典，告知 `tf.io.parse_single_example`函数每个 `tf.train.Example`数据项有哪些 Feature，以及这些 Feature 的类型、形状等属性。`tf.io.FixedLenFeature`有三个参数：**shape、dtype、default_value（可省略）**。因为这里的数据项都是单个的数值或者字符串，所以 shape 为空数组

运行以上代码后就获得了一个可用的 `tf.data.Dataset`对象了

### tf.function：图执行模式

在追求高性能或者部署模型时，希望使用 Graph 模式。TensorFlow 2 提供了 `tf.function`模块，结合 AutoGraph 机制，使得仅需要添加一个 `@tf.function`修饰符就可将被修饰部分以 Graph 模式运行

#### tf.function 基础使用方法

在 TensorFlow 2 中推荐使用 `tf.function`实现 Graph 执行模式。只需要将希望以 Graph 模式运行的代码封装在一个函数内，并在函数前加上 `@tf.function`修饰即可，如下例所示：

```python
num_batches = 1000
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()
model = CNN()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predections = model(images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        loss = tf.reduce_mean(loss)
        # 注意：这里使用的是tf.print()，@tf.function下不支持Python内置的print方法
        tf.print('loss', loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

for batch_index in range(num_batches):
    images, labels = data_loader.get_batch(batch_size)
    train_step(images, labels)
```

* 一般而言，**当被修饰函数由较多小的操作组成时**，`@tf.function`带来的性能提升效果较大。**而当被修饰函数的操作数量较少，但单一操作很耗时的时候**，则带来的性能提升不会太大
* **注意：**并非任何函数都可以被 `@tf.function`修饰。`@tf.function`使用静态编译将函数内的代码转换成计算图，因此对函数内可使用的语句有一定限制（仅支持 Python 的一个子集）。且需要函数内的操作本身能够被构建为计算图。**因此，建议在函数内只使用 TensorFlow 的原生操作，不要使用过于复杂的 Python 语句，函数参数只包括 TensorFlow 张量或 Numpy 数组**，并最好是能够按照计算图的思想去构建函数

#### Autograph 编码规范总结

* 被 `@tf.function`修饰的函数应尽可能使用 TensorFlow 中的函数，而不是 Python 中的其它函数，例如使用 `tf.print`而不是 `print`，使用 `tf.range`而不是 `range`，使用 `tf.constant(True)`而不是 `True`
* 避免在被 `@tf.function`修饰的函数内部定义 `tf.Variable`
* 被 `@tf.function`修饰的函数不可修改该函数外部的 Python 列表或字典等数据结构变量

#### 使用 tf.Module 封装 Autograph

在介绍 Autograph 编码规范时提到了在构建  Autograph 时应该避免在 `tf.function`修饰的函数内部定义 `tf.Variable`。但是如果在函数外部定义的话又会显得这个函数有外部变量依赖，封装不够完好。一种思路是定义一个类，并将相关的 `tf.Variable`创建放在类的初始化方法中，而将函数逻辑放在类的其它方法中

TensorFlow 提供了一个基类 `tf.Module`，通过继承它构建子类，不仅可以实现以上思路，而且可以非常方便的管理它引用的其它 Module，最重要的是，**可以利用 `tf.saved_model`保存模型从而实现跨平台部署**。实际上，`tf.keras.models.Model, tf.keras.layers.Layer`都继承自 `tf.Module`

```python
class DemoModule(tf.Module):
    def __init__(self, init_value=tf.constant(0.0), name=None):
        super().__init__(name=name)
        with self.name_scope:    #相当于 with tf.name_scope('demo_module')
            self.x = tf.Variable(init_value, dtype=tf.float32, trainable=True)
    
    # 在tf.function中用input_signature限定输入张量的签名类型：shape和dtype
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
    def addprint(self, a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print(self.x)
            return self.x
```



#### tf.function 的内在机制

当被 `@tf.function`修饰的函数 **第一次** 被调用时，进行以下操作：

* 在 Eager 模式关闭的环境下，函数内的代码依次运行。也就是说，每个 `tf.`方法都只是定义了计算节点，而并没有进行任何实质计算。这与 TensorFlow 1 的 Graph 模式是一致的
* 使用 AutoGraph 将函数中的 Python 控制流语句转换成 TensorFlow 计算图中的对应节点（比如 `while`和 `for`语句转换为 `tf.while`，`if`语句转换为 `tf.cond`等等）
* 基于上面两步，建立函数内代码的计算图表示（为了保证图的计算顺序，其中还会加入一些 `tf.control_dependencies`节点）
* 运行一次这个计算图
* 基于 **函数的名字和输入的参数类型生成一个哈希值**，并将建立的计算图缓存到一个哈希表中

在被 `@tf.function`修饰的函数之后再次被调用的时候，根据 **函数名和输入参数的类型** 计算哈希值，检查哈希表中是否已经有了对应计算图的缓存，如果是，则直接使用已缓存的计算图，否则重新按上述步骤建立计算图

为了厘清这个过程，思考以下代码段的输出结果：

```python
@tf.function
def f(x):
    print('The function is running in Python')
    tf.print(x)

a = tf.constant(1, dtype=tf.int32)
f(a)
b = tf.constant(2, dtype=tf.int32)
f(b)
b_ = np.array(2, dtype=np.int32)
f(b_)
c = tf.constant(0.1, dtype=tf.float32)
f(c)
d = tf.constant(0.2, dtype=tf.float32)
f(d)
```

答案是：

```python
The function is running in Python
1
2
2
The function is running in Python
0.1
0.2
```

当计算 `f(a)`时，由于是第一次调用该函数，TensorFlow 进行了以下操作：

1. 将函数内的代码依次运行了一遍（因此输出了文本）
2. 构建了计算图，然后运行了一次该计算图（因此输出了 1）。这了 `tf.print(x)`可以作为计算图的节点，但 Python 内置的 `print`则不能被转换成计算图的节点。所以计算图中只包含了 `tf.print(x)`这一操作
3. 将该计算图缓存tf.int32到了一个哈希表中（如果之后再有类型为 `tf.int32`，shape 为空的张量输入，则重复使用这个已经被构建出来的计算图）

值得注意的是在计算 `f(b_)`时，TensorFlow 自动将 Numpy 的数据结构转换成了 TensorFlow 中的张量，因此仍然能够复用之前已经构建的计算图

**而对于 `@tf.function`对 Python 内置的整数和浮点数类型的处理方式**，通过以下示例展示：

```python
@tf.function
def f(x):
    print('The function is running in Python')
    tf.print(x)

f(1)
f(2)
f(1)
f(0.1)
f(0.2)
f(0.1)
```

输出结果为：

```python
The function is running in Python
1
The function is running in Python
2
1
The function is running in Python
0.1
The function is running in Python
0.2
0.1
```

简而言之，对于 Python 内置的整数和浮点数类型，只有当值完全相同时，才会复用之前建立的计算图，而并不会自动将 Python 内置的整数或浮点数转换成张量。所以 **当函数参数包含 Python 内置整数或浮点数时要格外小心。一般而言，应当只在指定超参数等少数场合使用 Python 内置类型作为被 `@tf.function` 修饰的函数的参数**

#### AutoGraph：将 Python 控制流转换为 TensorFlow 计算图

以下是一个示例，使用 `tf.autograph`模块的低层 API `tf.autograph.to_code`将函数 `square_if_positive`转换成 TensorFlow 计算图：

```python
@tf.function
def square_if_positive(x):
    if x>0:
        x = x*x
    else:
        x = 0
    return x

print(tf.autograph.to_code(square_if_positive.python_function))
```

输出：

```python
def tf__square_if_positive(x):
    do_return = False
    retval_ = ag__.UndefinedReturnValue()
    cond = x > 0

    def get_state():
        return ()

    def set_state(_):
        pass

    def if_true():
        x_1, = x,
        x_1 = x_1 * x_1
        return x_1

    def if_false():
        x = 0
        return x
    x = ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
    do_return = True
    retval_ = x
    cond_1 = ag__.is_undefined_return(retval_)

    def get_state_1():
        return ()

    def set_state_1(_):
        pass

    def if_true_1():
        retval_ = None
        return retval_

    def if_false_1():
        return retval_
    retval_ = ag__.if_stmt(cond_1, if_true_1, if_false_1, get_state_1, set_state_1)
    return retval_
```

可以看到，原函数中的 Python 控制流 `if... else...`被转换为了 `x = ag__.if_stmt(cond, if_true, if_false, get_state, set_state)`这种计算图式的写法。AutoGraph 起到了类似编译器的作用，能够帮助用户通过自然的 Python 控制流构建计算图

### tf. TensorArray：TensorFlow 动态数组

在部分网络结构中，可能需要将一系列张量以数组的方式依次存放起来，以供进一步处理。在 Eager 模式下，可以直接使用一个 Python List 存放。但如果需要基于计算图的特性（例如使用 `@tf.function`加速模型运行或者使用 SavedModel 导出模型），就无法使用这种方式了。因此，TensorFlow 提供了 `tf.TensorArray`，一种支持计算图特性的 TensorFlow 动态数组

* 其声明方式为：`arr = tf.TensorArray(dtype, size, dynamic_size=False)`，如果将 dynamic_size 参数设为 True，则这个数组会自动增长空间
* 其读取和写入方法为：
  1. `write(index, value)`：将 value 写入数组的 Index 位置
  2. `read(index)`：读取数组 index 位置的值

除了上述基本操作之外，TensorArray 还具有 `stack()、unstack()`等常用操作，可以参考 [文档](https://tensorflow.google.cn/versions/r2.1/api_docs/python/tf/TensorArray?hl=en) 来了解详情

* **注意：**由于需要支持计算图，**`tf.TensorArray`的 `write()`方法是不可以忽略左值的！**也就是说，在 Graph 模式下，必须按照以下形式写入数组：`arr = arr.write(index, value)`，这个赋值是不可以省略的

### tf. config：GPU 的使用与分配

#### 指定当前程序使用的 GPU

首先，通过 `tf.config.list_physical_devices`，可以获得当前主机上某种特定运算设备类型（如 CPU 或 GPU）的列表，例如，在一台具有四块 GPU 和一颗 CPU 的工作站上运行以下代码：

```python
gpus = tf.config.list_physical_devices(device_type='GPU')
cpus = tf.config.list_physical_devices(device_type='CPU')
print(gpus, cpus)
```

输出：

```python
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
 PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'),
 PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'),
 PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU')]
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
```

然后，通过 `tf.config.set_visible_devices`，可以设置当前程序可见的设备范围（即使用范围）。例如，假设在上述 4 卡的机器中需要限定只使用标号为 0、1 的两块卡，可以添加以下代码：

```python
gpus = tf.config.list_physical_devices(device_type='GPU')
tf.config.set_visible_devices(devices=gpus[0:2], device_type='GPU')
```

当然，TensorFlow 1 时常用的方法也仍然可用：

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
```

#### 设置显存使用策略

默认情况下 TensorFlow 将使用几乎所有可用显存。不过，TensorFlow 提供两种显存使用策略：

1. 仅在需要时申请显存空间（程序初始运行时消耗很少的显存，随着程序的运行而动态申请显存）
2. 限制消耗固定大小的显存（程序不会超出限定的显存大小，超出则报错）

通过 `tf.config.experimental.set_memory_growth`将 GPU 的显存使用策略设置为 “仅在需要时申请显存空间”，以下代码将所有 GPU 设置为仅在需要时申请显存空间：

```python
gpus = tf.config.list_physical_devices(device_type='GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(device=g, enable=True)
```

以下代码通过 `tf.config.set_logical_device_configuration`方法并传入 `tf.config.LogicalDeviceConfiguration`实例，设置 TensorFlow 固定消耗 GPU:0 的 1 GB 显存（就是建立了一块显存大小为 1 GB 的虚拟 GPU）：

```python
gpus = tf.config.list_physical_devices(device_type='GPU')
tf.config.set_logical_device_configuration(gpus[0],
                                          [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
```

#### 单 GPU 模拟多 GPU 环境

当本地的开发环境仅有一块 GPU，但却需要编写多 GPU 的程序在工作站上进行训练任务时，TensorFlow 提供了一个功能，可以允许在本地开发环境中建立多个模拟 GPU，从而让多 GPU 程序的调试变得更加方便。以下代码在 physical device GPU:0 的基础上建立了两块显存均为 2 GB 的虚拟 GPU：

```python
gpus = tf.config.list_physical_devices(device_type='GPU')
tf.config.set_logical_device_configuration(gpus[0],
                                          [tf.config.LogicalDeviceConfiguration(memory_limit=2048),
                                           tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
```

只要在单机多卡训练的代码前加上以上代码，即可让原本为多 GPU 设计的代码在单 GPU 环境下正确运行

## TensorFlow 模型的导出

### 使用 SavedModel 完整导出模型

在部署模型时，第一步往往是将训练好的整个模型完整导出为一系列的标准格式文件，然后就可以在不同的平台上部署模型文件。TensorFlow 提供了 SavedModel 格式，该格式包含模型的完整信息，当模型导出为 SavedModel 文件后，无需建立模型的源代码即可再次运行模型

Keras 模型均可方便地导出为 SavedModel 格式。但需要注意的是，**因为 SavedModel 基于计算图，所以对于使用继承 `tf.keras.Model`类建立的 Keras 模型，其需要导出到 SavedModel 格式的方法（比如 `call`方法）都需要使用 `@tf.function`修饰。另外，保存的模型在使用 SavedModel 载入后将不能使用 `model()`直接进行推断，需要显式地调用 `model.call()`**

以下代码示例将一个名为 model 的 Keras 模型导出为 SavedModel：

```python
tf.saved_model.save(model, '保存的目标文件夹')
```

在需要载入 SavedModel 文件时：

```python
model = tf.saved_model.load('保存的目标文件夹')
```

## TensorFlow 分布式训练

针对不同使用场景，TensorFlow 在 `tf.distribute.Strategy`中提供了若干种分布式策略

### 单机多卡训练：MirroredStrategy

`tf.distribute.MirroredStrategy`是一种数据并行的 **同步式** 分布式策略，主要支持多块 GPU 在同一台主机上的训练。使用这种策略时，只需要实例化一个 `MirroredStrategy`策略，并将模型构建的代码放入 `strategy.scope()`的上下文环境中：

```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = model_creation()
    model.compile()
model.fit()
```

* 在策略参数中指定设备，如：

  `strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])`

MirroredStrategy 的步骤如下：

1. 训练开始前，该策略在所有 N 个计算设备上均各复制一份完整的模型
2. 每次训练传入一个批次的数据时，将数据分成 N 份，分别传入 N 个计算设备
3. N 个计算设备使用本地变量（镜像变量）分别计算自己所获得的部分数据的梯度
4. 使用分布式计算的 All-reduce 操作，在计算设备间高效交换梯度数据并进行求和，使得最终每个设备都拥有了所有设备的梯度之和
5. 使用梯度求和结果更新本地变量（镜像变量）
6. 当所有设备均更新本地变量后，进行下一轮训练（即该并行策略是同步的）

默认情况下，TensorFlow 中的 MirroredStrategy 策略使用 NVIDIA NCCL 进行 All-reduce 操作