import tvm
from tvm import relay
from tvm.relay import analysis
from tvm.contrib.download import download_testdata
import onnx

from count_flops_op import count_flops_op_map

# Load the ONNX model
model_url = (
    "https://github.com/onnx/models/raw/main/"
    "vision/classification/resnet/model/"
    "resnet18-v1-7.onnx"
)
model_path = download_testdata(model_url, "resnet18-v1-7.onnx", module="onnx")
onnx_model = onnx.load(model_path)
shape_dict = {"data": (1, 3, 224, 224)}

# Convert the ONNX model to a Relay function
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)





# Calculate FLOPs for resnet18
flops_dict = {}
def count_flops(expr):
    if isinstance(expr, relay.Call):
        flops = 0
        if expr.op.name in count_flops_op_map:
            flops = count_flops_op_map[expr.op.name](expr)
        else:
            print(f"Unknown operator: {expr.op.name}")
        if flops_dict.get(expr.op.name) is None:
            flops_dict[expr.op.name] = flops
        else:
            flops_dict[expr.op.name] += flops

    
analysis.post_order_visit(mod['main'], count_flops)

for key, value in flops_dict.items():
    print(key, value)
