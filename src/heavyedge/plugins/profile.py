"""Commands to process profiles."""

import pathlib
from importlib.metadata import entry_points

from heavyedge.cli.command import Command, register_command

PLUGIN_ORDER = 0.0


@register_command("prep", "Preprocess raw profiles")
class PrepCommand(Command):
    def add_parser(self, main_parser):
        prep = main_parser.add_parser(
            self.name,
            description="Preprocess raw profiles and save as hdf5 file.",
            epilog="The resulting hdf5 file is in 'ProfileData' structure.",
        )
        prep.add_argument(
            "--type",
            choices=[ep.name for ep in entry_points(group="heavyedge.rawdata")],
            required=True,
            help="Type of the raw profile data.",
        )
        prep.add_argument("--name", help="Name to label output dataset.")
        prep.add_argument(
            "raw",
            type=pathlib.Path,
            help="Path to raw profile data.",
        )
        prep.add_config_argument(
            "--res",
            type=float,
            help=(
                "Spatial resolution of profile data, i.e., points per unit length. "
                "Length unit must match the profile height unit."
            ),
        )
        prep.add_config_argument(
            "--std",
            type=float,
            help="Standard deviation of Gaussian kernel.",
        )
        prep.add_config_argument(
            "--std-thres",
            type=float,
            help="Standard deviation threshold for contact point detection.",
        )
        prep.add_argument("-o", "--output", type=pathlib.Path, help="Output file path")

    def run(self, args):
        import numpy as np

        from heavyedge.api import preprocess
        from heavyedge.io import ProfileData

        self.logger.info(f"Preprocessing: {args.raw}")

        raw_type = entry_points(group="heavyedge.rawdata")[args.type].load()
        raw = raw_type(args.raw)
        raw_profiles = zip(raw.profiles(), raw.profile_names())

        # search for the first valid profile
        for profile, name in raw_profiles:
            if len(profile) == 0:
                continue
            if np.any(np.isnan(profile)):
                continue
            M = len(profile)
            break

        with ProfileData(args.output, "w").create(M, args.res, args.name) as out:
            # write the first valid profile
            Y, L = preprocess(profile, args.std, args.std_thres)
            out.write_profiles(Y.reshape(1, -1), [L], [name])

            for profile, name in raw_profiles:
                if len(profile) == 0:
                    continue
                if np.any(np.isnan(profile)):
                    continue
                Y, L = preprocess(profile, args.std, args.std_thres)
                out.write_profiles(Y.reshape(1, -1), [L], [name])

        self.logger.info(f"Preprocessed: {out.path}")


@register_command("mean", "Compute mean profile")
class MeanCommand(Command):
    def add_parser(self, main_parser):
        mean = main_parser.add_parser(
            self.name,
            description="Compute mean profile and save as hdf5 file.",
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
        mean.add_argument("-o", "--output", type=pathlib.Path, help="Output file path")

    def run(self, args):
        import numpy as np

        from heavyedge.api import mean
        from heavyedge.io import ProfileData

        self.logger.info(f"Averaging: {args.profiles}")

        with ProfileData(args.profiles) as data:
            _, M = data.shape()
            res = data.resolution()
            name = data.name()
            pmean = mean(data.profiles(), args.wnum)

        with ProfileData(args.output, "w").create(M, res, name) as out:
            L = len(mean)
            pmean = np.pad(pmean, (0, M - L), constant_values=np.nan)
            out.write_profile(pmean.reshape(1, -1), [L], [name])

        self.logger.info(f"Averaged: {out.path}")
