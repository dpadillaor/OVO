"""Command line: ``python -m ovo <command> ...``.

    python -m ovo eval instances --dataset Replica --experiment my_exp [--scenes office0 room0]

Commands only read an experiment's outputs; running OVO itself is still ``run_eval.py``.
"""

import argparse
import json

from ovo import layout


def _eval_instances(args) -> None:
    from ovo.eval import benchmarks, instances

    exp = layout.ExperimentLayout.named(args.dataset, args.experiment)
    scenes = args.scenes or sorted(p.name.removesuffix(".classes.json") for p in exp.instance_pred.glob("*.classes.json"))
    result = instances.evaluate_experiment(
        exp, scenes, benchmarks.for_dataset(args.dataset), layout.instance_gt_dir(args.dataset)
    )
    print(json.dumps(instances.nan_to_none(result["all"]), indent=2))
    print(f"Written to {exp.instance_ap}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ovo")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser("eval", help="evaluate an experiment").add_subparsers(dest="what", required=True)
    ins = evaluate.add_parser("instances", help="instance AP (class-aware and class-agnostic)")
    ins.add_argument("--dataset", required=True, help="Replica or ScanNet")
    ins.add_argument("--experiment", required=True)
    ins.add_argument("--scenes", nargs="+", default=None, help="default: every scene with predictions")
    ins.set_defaults(run=_eval_instances)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
