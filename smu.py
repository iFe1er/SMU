# coding=utf-8

import tensorflow as tf

def SMU(x,alpha=0.25):
    mu = tf.compat.v1.get_variable('SMU_mu', shape=(),
                       initializer=tf.constant_initializer(1000000),
                       dtype=tf.float32)
    return ((1+alpha)*x + (1-alpha)*x*tf.math.erf(mu*(1-alpha)*x))/2

def SMU1(x,alpha=0.25):
    mu = tf.compat.v1.get_variable('SMU1_mu', shape=(),
                       initializer=tf.constant_initializer(4.352665993287951e-9),
                       dtype=tf.float32)
    return ((1+alpha)*x+tf.math.sqrt(tf.math.square(x-alpha*x)+tf.math.square(mu)))/2
    
def test_SMU(x):
    print(SMU(x))
    
def test_SMU1(x):
    print(SMU1(x))

def test():
    x = tf.convert_to_tensor(np.array([[-0.6],[0.6]]),dtype=tf.float32)
    test_SMU(x)
    test_SMU1(x)

if __name__ == '__main__':
    test()
