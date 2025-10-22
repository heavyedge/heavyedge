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
            "--sigma",
            type=float,
            help="Standard deviation of Gaussian kernel.",
        )
        prep.add_config_argument(
            "--std-thres",
            type=float,
            help="Standard deviation threshold for contact point detection.",
        )
        prep.add_config_argument(
            "--fill-value",
            choices=["0", "nan"],
            help="Value to fill profile after the contact point.",
        )
        prep.add_config_argument(
            "--batch-size",
            type=int,
            help="Batch size to load data. If not provided, load entire profiles.",
        )
        prep.add_argument("-o", "--output", type=pathlib.Path, help="Output file path")

    def run(self, args):
        import numpy as np

        from heavyedge.api import fill_after, preprocess
        from heavyedge.io import ProfileData

        def is_invalid(profile):
            return (len(profile) == 0) or np.any(np.isnan(profile))

        def save_batch(Ys, Ls, names, fill_value, outfile):
            Ys, Ls = np.array(Ys), np.array(Ls)
            if fill_value is not None:
                fill_after(Ys, Ls, fill_value)
            outfile.write_profiles(Ys, Ls, names)

        self.logger.info(f"Preprocessing: {args.raw}")

        raw_type = entry_points(group="heavyedge.rawdata")[args.type].load()
        raw = raw_type(args.raw)
        raw_profiles = zip(raw.profiles(), raw.profile_names())

        if args.fill_value == "0":
            args.fill_value = 0

        # search for the first valid profile to determine M
        for profile, name in raw_profiles:
            if is_invalid(profile):
                continue
            M = len(profile)
            break
        else:
            raise ValueError(f"No valid profile in {args.raw}.")
        Y, L = preprocess(profile, args.sigma, args.std_thres)
        Ys, Ls, names = [Y], [L], [name]
        count = 1

        with ProfileData(args.output, "w").create(M, args.res, args.name) as out:
            # iterate over the remaining profiles
            for profile, name in raw_profiles:
                if is_invalid(profile):
                    continue
                Y, L = preprocess(profile, args.sigma, args.std_thres)
                Ys.append(Y)
                Ls.append(L)
                names.append(name)
                count += 1

                if count == args.batch_size:
                    save_batch(Ys, Ls, names, args.fill_value, out)
                    Ys, Ls, names = [], [], []
                    count = 0

            # save remaining batch
            save_batch(Ys, Ls, names, args.fill_value, out)

        self.logger.info(f"Preprocessed: {out.path}")


@register_command("outlier", "Filter outlier profiles")
class OutlierCommand(Command):
    def add_parser(self, main_parser):
        outlier = main_parser.add_parser(
            self.name,
            description="Remove outlier profiles and save as hdf5 file.",
            epilog="The resulting hdf5 file is in 'ProfileData' structure.",
        )
        outlier.add_argument(
            "profiles",
            type=pathlib.Path,
            help="Path to preprocessed profile data in 'ProfileData' structure.",
        )
        outlier.add_config_argument(
            "--z",
            type=float,
            help="Modified Z-score threshold for outlier detection.",
        )
        outlier.add_argument(
            "-o", "--output", type=pathlib.Path, help="Output file path"
        )

    def run(self, args):
        from heavyedge import ProfileData
        from heavyedge.api import outlier

        self.logger.info(f"Removing outliers: {args.profiles}")

        with ProfileData(args.profiles) as data:
            _, M = data.shape()
            res = data.resolution()
            name = data.name()
            is_outlier = outlier(data.profiles(), args.z)

            with ProfileData(args.output, "w").create(M, res, name) as out:
                for skip, Y, L, name in zip(
                    is_outlier,
                    data._file["profiles"],
                    data._file["len"],
                    data.profile_names(),
                ):
                    if not skip:
                        out.write_profiles(Y.reshape(1, -1), [L], [name])

        self.logger.info(f"Removed outliers: {out.path}")


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
            x = data.x()
            res = data.resolution()
            name = data.name()
            pmean = mean(x, data.profiles(), args.wnum)

        with ProfileData(args.output, "w").create(M, res, name) as out:
            L = len(pmean)
            pmean = np.pad(pmean, (0, M - L), constant_values=np.nan)
            out.write_profiles(pmean.reshape(1, -1), [L], [name])

        self.logger.info(f"Averaged: {out.path}")


@register_command("merge", "Merge profile data")
class MergeCommand(Command):
    def add_parser(self, main_parser):
        merge = main_parser.add_parser(
            self.name,
            description="Merge profile data and save as hdf5 file.",
            epilog="The resulting hdf5 file is in 'ProfileData' structure.",
        )
        merge.add_argument(
            "profiles",
            nargs="+",
            type=pathlib.Path,
            help="Paths to preprocessed profile data in 'ProfileData' structure.",
        )
        merge.add_argument("--name", help="Name to label output dataset.")
        merge.add_argument("-o", "--output", type=pathlib.Path, help="Output file path")

    def run(self, args):
        from heavyedge.io import ProfileData

        with ProfileData(args.profiles[0]) as data:
            _, M = data.shape()
            res = data.resolution()

        with ProfileData(args.output, "w").create(M, res, args.name) as out:
            for p in args.profiles:
                with ProfileData(p) as data:
                    out.write_profiles(
                        data._file["profiles"][:],
                        data._file["len"][:],
                        data._file["names"][:],
                    )
