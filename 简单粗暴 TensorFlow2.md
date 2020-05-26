## 简单粗暴 TensorFlow2

### TF2  基础

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

  

### TF2 模型建立与训练

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
  gradients = loss.gradient(loss, model.trainable_variables)
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

### TF2 常用模块

#### tf.train.Checkpoint

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

#### TensorBoard：训练过程可视化

##### 实时查看参数变化情况

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

##### 查看 Graph 和 Profile 信息

* 还可以在训练开始前通过使用 `tf.summary.trace_on`来开启 Trace，之后 TensorFlow 会记录下训练时的大量信息（如计算图结构、每个 operation 的耗时等）。在训练完成后，使用 `tf.summary.trace_export`将记录结果输出至文件

  ```python
  tf.summary.trace_on(graph=True, profiler=True)  # 开启trace，可以选择是否要记录图结构和profile信息
  '''
  进行训练
  '''
  with summary_writer.as_default():
      tf.summary.trace_export(name='model_trace', step=0, profiler_outdir=log_dir)
  ```

#### tf.data：数据集的创建与预处理

##### 数据集对象的建立

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

##### 数据集对象的预处理

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

##### tf.data 的并行化

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

##### Dataset 元素的获取与使用

`tf.data.Dataset`类对象本身是一个 Python 的 **可迭代对象**，因此可以使用 For 循环迭代：`for a, b, c in dataset`

* `iter()`：也可以使用 `iter()`显示地创建一个 Python 迭代器并使用 `next()`方法获取下一元素：

  ```python
  dataset = tf.data.Dataset.from_tensor_slices((A, B, C))
  it = iter(dataset)
  a0, b0, c0 = next(it)
  a1, b1, c1 = next(it)
  ```

##### cats_vs_dogs 图像分类实例

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

