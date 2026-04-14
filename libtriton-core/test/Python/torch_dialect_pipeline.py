from libtriton_core import ir, passmanager
from libtriton.core.dialects import func, torch


def main():
    ctx = ir.Context()
    torch.register_dialect(ctx)

    with ctx, ir.Location.unknown():
        module = ir.Module.create()
        vtensor = ir.Type.parse("!torch.vtensor<[1],f32>")
        func_type = ir.FunctionType.get([vtensor], [vtensor])

        function = func.FuncOp("main", func_type)
        module.body.append(function.operation)
        entry = ir.Block.create_at_start(function.regions[0], [vtensor])

        with ir.InsertionPoint(entry):
            relu = torch.AtenReluOp(result=vtensor, self_=entry.arguments[0])
            func.ReturnOp([relu.result])

        passmanager.PassManager.parse(
            "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)"
        ).run(module.operation)


if __name__ == "__main__":
    main()
