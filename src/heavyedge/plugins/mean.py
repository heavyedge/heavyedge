"""Commands to average profiles."""

import pathlib

from heavyedge.cli.command import Command, register_command

PLUGIN_ORDER = 0.1


@register_command("mean", "Compute mean profile")
class MeanCommand(Command):
    def add_parser(self, main_parser):
        mean = main_parser.add_parser(
            self.name,
            description="Compute Wasserstein mean profile and save as hdf5 file.",
            epilog="The resulting hdf5 file is in 'ProfileData' structure.",
        )
        mean.add_argument(
            "profiles",
            type=pathlib.Path,
            help="Path to preprocessed profile data in 'ProfileData' structure.",
        )
        mean.add_config_argument(
            "--wnum",
            type=int,
            help="Number of sample points to compute Wasserstein mean.",
        )
        mean.add_argument(
            "--batch-size",
            type=int,
            help="Batch size to load data. If not provided, loads entire profiles.",
        )
        mean.add_argument(
            "-o", "--output", type=pathlib.Path, help="Output npy file path"
        )

    def run(self, args):
        import numpy as np

        from heavyedge import ProfileData
        from heavyedge.api import mean_wasserstein

        out = args.output.expanduser()

        self.logger.info(f"Estimating Wasserstein mean profile: {out}")

        def logger(msg):
            self.logger.info(f"{out} : {msg}")

        with ProfileData(args.profiles) as file:
            _, M = file.shape()
            res = file.resolution()
            name = file.name()
            mean = mean_wasserstein(file, args.wnum, args.batch_size, logger)

        with ProfileData(args.output, "w").create(M, res, name) as out:
            L = len(mean)
            mean = np.pad(mean, (0, M - L), constant_values=np.nan)
            out.write_profiles(mean.reshape(1, -1), [L], [name])

        self.logger.info(f"Saved {out}.")
