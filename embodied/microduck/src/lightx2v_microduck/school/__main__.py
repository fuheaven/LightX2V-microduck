import argparse
import json
import os


def main(argv=None):
    parser = argparse.ArgumentParser(description="MicroDuck School: frozen official locomotion, CEM task learning")
    commands = parser.add_subparsers(dest="action", required=True)
    preview = commands.add_parser("preview")
    preview.add_argument("instruction")
    worker = commands.add_parser("worker")
    worker.add_argument("request")
    train = commands.add_parser("train")
    train.add_argument("--instruction", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--budget-s", type=int, default=600)
    train.add_argument("--generations", type=int, default=20)
    train.add_argument("--population", type=int, default=8)
    train.add_argument("--train-count", type=int, default=3)
    train.add_argument("--test-count", type=int, default=10)
    train.add_argument("--resume")
    for name in ("evaluate", "live"):
        p = commands.add_parser(name)
        p.add_argument("--artifact", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--count", type=int, default=10)
        p.add_argument("--budget-s", type=int, default=600)
    args = parser.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    from .courses import Course, parse_lesson
    from .learning import execute_skill, train

    if args.action == "preview":
        result = parse_lesson(args.instruction)
    elif args.action == "worker":
        from .service import worker

        return worker(args.request)
    elif args.action == "train":
        course = Course.from_dict(parse_lesson(args.instruction)["course"])
        result = train(
            course,
            args.output,
            seed=args.seed,
            budget_s=args.budget_s,
            generations=args.generations,
            population=args.population,
            train_count=args.train_count,
            test_count=args.test_count,
            resume=args.resume,
        )
    else:
        result = execute_skill(args.artifact, args.output, seed=args.seed, live=args.action == "live", count=args.count, budget_s=args.budget_s)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
