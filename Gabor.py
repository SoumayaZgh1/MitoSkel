class GaborLayer(layers.Layer):
    def __init__(self, sigma=0.1, theta=0.0, lambd=5.0, gamma=0.5, kernel_size=3):
        super(GaborLayer, self).__init__()
        self.sigma = tf.Variable(sigma, trainable=True)
        self.theta = tf.Variable(theta, trainable=True)
        self.lambd = tf.Variable(lambd, trainable=True)
        self.gamma = tf.Variable(gamma, trainable=True)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        half_size = (self.kernel_size - 1) // 2
        x, y = tf.meshgrid(tf.range(-half_size, half_size + 1), tf.range(-half_size, half_size + 1))
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        self.x = x
        self.y = y

    def call(self, inputs):
        channel_outputs = []
        for i in range(inputs.shape[-1]):
            single_channel = inputs[..., i:i+1]
            
            x_0 = self.x * tf.cos(self.theta) + self.y * tf.sin(self.theta)
            y_0 = -self.x * tf.sin(self.theta) + self.y * tf.cos(self.theta)
            gabor_filter = tf.exp(-(x_0**2 + (self.gamma**2) * y_0**2) / (2 * self.sigma**2)) * tf.cos(2 * np.pi * x_0 / self.lambd)
            gabor_filter = gabor_filter / (2 * np.pi * self.sigma**2)
            gabor_filter = tf.expand_dims(gabor_filter, axis=-1)
            gabor_filter = tf.expand_dims(gabor_filter, axis=-1)

            filtered = tf.nn.conv2d(single_channel, gabor_filter, strides=[1, 1, 1, 1], padding='SAME')
            channel_outputs.append(filtered)

        return tf.concat(channel_outputs, axis=-1)

    def get_config(self):
        config = super(GaborLayer, self).get_config()
        config.update({
            "sigma": float(self.sigma.numpy()),  # Ensure conversion to float
            "theta": float(self.theta.numpy()),
            "lambd": float(self.lambd.numpy()),
            "gamma": float(self.gamma.numpy()),
            "kernel_size": self.kernel_size
        })
        return config


class TAM(layers.Layer):
    def __init__(self, input_channels, gate_channels):
        super(AttentionGate, self).__init__()
        self.input_channels = input_channels
        self.gate_channels = gate_channels
        self.W_g = layers.Conv2D(gate_channels, (1, 1), padding='same')
        self.W_x = layers.Conv2D(gate_channels, (1, 1), padding='same')
        self.psi = layers.Conv2D(1, (1, 1), padding='same')
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x, g):
        x1 = self.W_g(g)
        x2 = self.W_x(x)
        psi = self.sigmoid(self.psi(tf.nn.relu(x1 + x2)))
        return x * psi

    def get_config(self):
        config = super(TAM, self).get_config()
        config.update({
            "input_channels": self.input_channels,
            "gate_channels": self.gate_channels
        })
        return config


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, x):
        attn_output = self.att(x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim
        })
        return config


def double_conv_block(x, filters, use_gabor=False):
    if use_gabor:
        x = GaborLayer()(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x


def upsampling_block(x, skip, filters, use_attention=True):
    x = layers.UpSampling2D((2, 2))(x)
    if use_attention:
        skip = TAM(filters, filters // 2)(skip, x)
    x = layers.Concatenate()([x, skip])
    return double_conv_block(x, filters)

def GAU_Net(input_shape=(256, 256, 3), use_gabor=True, use_attention=True, use_transformer=False):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    e1 = double_conv_block(inputs, 64, use_gabor=use_gabor)
    p1 = layers.MaxPooling2D((2, 2))(e1)
    e2 = double_conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(e2)
    e3 = double_conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(e3)
    e4 = double_conv_block(p3, 512)
    
    # Bottleneck + Transformer
    if use_transformer:
        b = TransformerBlock(512, num_heads=8, ff_dim=2048)(e4)
    else:
        b = double_conv_block(e4, 512)
    
    # Decoder
    d1 = upsampling_block(b, e3, 256, use_attention)
    d2 = upsampling_block(d1, e2, 128, use_attention)
    d3 = upsampling_block(d2, e1, 64, use_attention)
    
    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d3)
    return Model(inputs, outputs)
