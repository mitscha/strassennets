import numpy as np
import mxnet as mx

class TerQuant(mx.operator.CustomOp):
    def __init__(self, scale):
        self._scale = float(scale)

    def forward(self, is_train, req, in_data, out_data, aux):
        full_data = in_data[0]
        # Quatize input NDarray to {-1,0,1}
        quant_data = self._scale*mx.nd.greater(full_data, 0.5*self._scale*mx.nd.ones(full_data.shape, full_data.context))\
            - self._scale*mx.nd.lesser(full_data, (-0.5)*self._scale*mx.nd.ones(full_data.shape, full_data.context))

        self.assign(out_data[0], req[0], quant_data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # Pass gradient through (straight-through estimator)
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("ter_quant")
class TerQuantProp(mx.operator.CustomOpProp):
    def __init__(self, scale=1.0):
        super(TerQuantProp, self).__init__(True)
        self._scale = scale

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        output_shape = data_shape
        return (data_shape,), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return TerQuant(self._scale)


class TerQuantRescaled(mx.operator.CustomOp):
    def __init__(self, scale):
        self._scale = float(scale)

    def forward(self, is_train, req, in_data, out_data, aux):
        full_data = in_data[0]

        thresh = mx.nd.mean(mx.nd.abs(full_data), axis=()).asnumpy()[0] * 0.7

        quant_data = mx.nd.greater(full_data, thresh*mx.nd.ones(full_data.shape, full_data.context))\
            - mx.nd.lesser(full_data, (-1.0)*thresh*mx.nd.ones(full_data.shape, full_data.context))

        scale = mx.nd.sum(mx.nd.multiply(full_data, quant_data), axis=()).asnumpy()[0]\
            / mx.nd.sum(mx.nd.abs(quant_data), axis=()).asnumpy()[0] * self._scale

        quant_data = scale * quant_data
        self.assign(out_data[0], req[0], quant_data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # Pass gradient through (straight-through estimator)
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("ter_quant_rescaled")
class TerQuantRescaledProp(mx.operator.CustomOpProp):
    def __init__(self, scale=1.0):
        super(TerQuantRescaledProp, self).__init__(True)
        self._scale = scale

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        output_shape = data_shape
        return (data_shape,), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return TerQuantRescaled(self._scale)
