import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("MatMul")
#@tf_func(tf.matmul)
class MatMul(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
      return [cls.make_tensor_from_onnx_node(node, tf.matmul, **kwargs)]

  @classmethod
  def version_9(cls, node, **kwargs):
    #return [cls.make_tensor_from_onnx_node(node, **kwargs)]
    tensor_dict = kwargs["tensor_dict"]
    def batch_matmul(x,y,
            transpose_a=False,
            transpose_b=False,
            adjoint_a=False,
            adjoint_b=False,
            a_is_sparse=False,
            b_is_sparse=False,
            name=None):
        def broad(a,b):
            return tf.concat([tf.shape(b)[0:-2],tf.shape(a)],axis=-1)
        def ori(a):
            return tf.shape(a)
        broadx = lambda: broad(x,y)
        broady = lambda: broad(y,x)
        orix = lambda: ori(x)
        oriy = lambda: ori(y)
        x_shape = tf.cond(tf.less(tf.rank(x),tf.rank(y)),broadx,orix)
        y_shape = tf.cond(tf.less(tf.rank(y),tf.rank(x)),broady,oriy)
        a = tf.broadcast_to(x,x_shape)
        b = tf.broadcast_to(y,y_shape)
        return tf.matmul(a,b,transpose_a,transpose_b,adjoint_a,adjoint_b,a_is_sparse,b_is_sparse,name)
    return [cls.make_tensor_from_onnx_node(node,batch_matmul, **kwargs)]
