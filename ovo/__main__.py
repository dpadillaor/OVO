"""Command line: ``python -m ovo <command> ...``.

    python -m ovo eval instances --dataset Replica --experiment my_exp [--scenes office0 room0]
    python -m ovo replay --dataset Replica --experiment my_exp --name aabb [--fusion aabb.yaml]
                         [--scenes office0 [--frame 1990]]

Commands work on an experiment's outputs; running OVO itself is still ``run_eval.py``.
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


def _replay(args) -> None:
    from ovo.replay.run import replay_experiment
    from ovo.utils import io_utils

    fusion = io_utils.load_config(args.fusion, inherit=False) if args.fusion else None
    out = replay_experiment(args.dataset, args.experiment, args.name, fusion, args.scenes, args.frame, args.device)
    print(f"Written to {out.root}; evaluate it with --experiment {args.experiment}/replays/{args.name}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ovo")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser("eval", help="evaluate an experiment").add_subparsers(dest="what", required=True)
    ins = evaluate.add_parser("instances", help="instance AP (class-aware and class-agnostic)")
    ins.add_argument("--dataset", required=True, help="Replica or ScanNet")
    ins.add_argument("--experiment", required=True)
    ins.add_argument("--scenes", nargs="+", default=None, help="default: every scene with predictions")
    ins.set_defaults(run=_eval_instances)

    rep = commands.add_parser("replay", help="re-run the fusion of an experiment from its pre-fusion checkpoints")
    rep.add_argument("--dataset", required=True)
    rep.add_argument("--experiment", required=True, help="the experiment that saved the checkpoints")
    rep.add_argument("--name", required=True, help="output: {experiment}/replays/{name}")
    rep.add_argument("--fusion", default=None, help="YAML shaped like the `semantic.instance_fusion` block (e.g. `accept: {min_cos: 0.8}`), "
                     "replacing it whole; default: the original")
    rep.add_argument("--scenes", nargs="+", default=None, help="default: every scene with checkpoints")
    rep.add_argument("--frame", type=int, default=None, help="checkpoint frame (one scene only); default: the last")
    rep.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    rep.set_defaults(run=_replay)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
