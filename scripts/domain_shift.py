import argparse
import shutil
import pickle
import logging
from omegaconf import OmegaConf
import re
import random
from pydantic import BaseModel
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_parser():
    parser = argparse.ArgumentParser(description="Generate a domain shift dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--full_candidate_subsets_path",
        type=str,
        required=True,
        help="Path to full-candidate-subsets.pkl",
    )
    parser.add_argument(
        "--visual_genome_images_dir",
        type=str,
        required=True,
        help="Path to VisualGenome images directory allImages/images",
    )
    return parser


def get_ms_domain_name(obj: str, context: str) -> str:
    return f"{obj}({context})"


class DataSplits(BaseModel):
    train: dict[str, list[str]]
    test: dict[str, list[str]]


class MetashiftData(BaseModel):
    selected_classes: list[str]
    spurious_class: str
    train_context: str
    test_context: str
    data_splits: DataSplits


class MetashiftFactory(object):
    object_context_to_id: dict[str, list[int]]
    visual_genome_images_dir: str

    def __init__(
        self,
        full_candidate_subsets_path: str,
        visual_genome_images_dir: str,
    ):
        """
        full_candidate_subsets_path: Path to `full-candidate-subsets.pkl`
        visual_genome_images_dir: Path to VisualGenome images directory `allImages/images`
        """
        with open(full_candidate_subsets_path, "rb") as f:
            self.object_context_to_id = pickle.load(f)
        self.visual_genome_images_dir = visual_genome_images_dir

    def _get_all_image_ids_with_object(self, obj: str) -> set[str]:
        """Get all image ids with given object and any context.
        Example:
            - get_all_image_ids_with_object(table) => [id~table(dog), id~table(cat), ...]
            - where id~domain, means an image sampled from the given domain.
        """
        return {
            key
            for key in self.object_context_to_id.keys()
            if re.match(f"^{obj}(.*)$", key)
        }

    def _get_image_ids(self, obj: str, context: str | None) -> set[str]:
        """Get image ids for the domain `obj(context)`."""
        if context is None:
            keys = self._get_all_image_ids_with_object(obj)
            ids = set()
            for key in keys:
                ids.update(self.object_context_to_id[key])
            return ids
        else:
            return self.object_context_to_id[get_ms_domain_name(obj, context)]

    def _get_class_domains(
        self, domains_specification: dict[str, tuple[str, str | None]]
    ) -> dict[str, tuple[list[str], list[str]]]:
        """Get train and test image ids for the given domains specification."""
        domain_ids = dict()
        for cls, (train_context, test_context) in domains_specification.items():
            if train_context == test_context:
                ids = self._get_image_ids(cls, train_context)
                domain_ids[cls] = [ids, ids]
                logger.info(
                    f"{get_ms_domain_name(cls, train_context or '*')}: {len(ids)}"
                    " -> "
                    f"{get_ms_domain_name(cls, test_context or '*')}: {len(ids)}"
                )
            else:
                train_ids = self._get_image_ids(cls, train_context)
                test_ids = self._get_image_ids(cls, test_context)
                domain_ids[cls] = [train_ids, test_ids]
                logger.info(
                    f"{get_ms_domain_name(cls, train_context or '*')}: {len(train_ids)}"
                    " -> "
                    f"{get_ms_domain_name(cls, test_context or '*')}: {len(test_ids)}"
                )
        return domain_ids

    def _sample_from_domains(
        self,
        seed: int,
        domains: dict[str, tuple[list[str], list[str]]],
        num_train_images_per_class: int,
        num_test_images_per_class: int,
    ) -> dict[str, tuple[list[str], list[str]]]:
        """Return sampled domain data from the given full domains."""
        # TODO: Do we have to ensure that there's no overlap between classes?
        # For example, we could have repeated files in training for different classes.
        sampled_domains = dict()
        for cls, (train_ids, test_ids) in domains.items():
            try:
                sampled_train_ids = random.Random(seed).sample(
                    list(train_ids), num_train_images_per_class
                )
                test_ids = test_ids - set(sampled_train_ids)
                sampled_test_ids = random.Random(seed).sample(
                    list(test_ids), num_test_images_per_class
                )
            except ValueError:
                logger.error(
                    f"{cls}: {len(train_ids)} train images, {len(test_ids)} test images"
                )
                raise Exception("Not enough images for this class")
            sampled_domains[cls] = (sampled_train_ids, sampled_test_ids)
        return sampled_domains

    def create(
        self,
        seed: int,
        selected_classes: list[str],
        spurious_class: str,
        train_spurious_context: str,
        test_spurious_context: str,
        num_train_images_per_class: int,
        num_test_images_per_class: int,
    ) -> MetashiftData:
        """Return (metadata, data) splits for the given data shift."""
        domains_specification = {
            **{cls: (None, None) for cls in selected_classes},
            spurious_class: (
                train_spurious_context,
                test_spurious_context,
            ),  # overwrite spurious_class
        }
        domains = self._get_class_domains(domains_specification)
        sampled_domains = self._sample_from_domains(
            seed=seed,
            domains=domains,
            num_train_images_per_class=num_train_images_per_class,
            num_test_images_per_class=num_test_images_per_class,
        )
        data_splits = {"train": dict(), "test": dict()}
        for cls, (train_ids, test_ids) in sampled_domains.items():
            data_splits["train"][cls] = train_ids
            data_splits["test"][cls] = test_ids

        return MetashiftData(
            selected_classes=selected_classes,
            spurious_class=spurious_class,
            train_context=train_spurious_context,
            test_context=test_spurious_context,
            data_splits=DataSplits(
                train=data_splits["train"],
                test=data_splits["test"],
            ),
        )

    def save_all(self, out_dir: str, info: dict[str, MetashiftData]):
        """Save all datasets to the given directory. """
        out_path = Path(out_dir)
        data_path = out_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        unique_ids = set()
        for dataset_name, data in info.items():
            for cls, ids in data.data_splits.train.items():
                unique_ids.update(ids)
                data.data_splits.train[cls] = [
                    str(data_path / f"{_id}.jpg") for _id in ids
                ]
            for cls, ids in data.data_splits.test.items():
                unique_ids.update(ids)
                data.data_splits.test[cls] = [
                    str(data_path / f"{_id}.jpg") for _id in ids
                ]
            with open(out_path / f"{dataset_name}.json", "w") as f:
                f.write(data.model_dump_json(indent=2))

        # Copy all images
        for _id in unique_ids:
            shutil.copy(
                Path(self.visual_genome_images_dir) / f"{_id}.jpg",
                data_path / f"{_id}.jpg",
            )


def get_dataset_name(task_name: str, experiment_name: str) -> str:
    return f"{task_name}_{experiment_name}"


def main():
    parser = setup_parser()
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    metashift_factory = MetashiftFactory(
        full_candidate_subsets_path=args.full_candidate_subsets_path,
        visual_genome_images_dir=args.visual_genome_images_dir,
    )
    info: dict[str, MetashiftData] = dict()
    for task_config in config.tasks:
        for experiment_config in task_config.experiments:
            data = metashift_factory.create(
                seed=task_config.seed,
                selected_classes=task_config.selected_classes,
                spurious_class=experiment_config.spurious_class,
                train_spurious_context=experiment_config.train_context,
                test_spurious_context=experiment_config.test_context,
                num_test_images_per_class=task_config.num_images_per_class_test,
                num_train_images_per_class=task_config.num_images_per_class_train,
            )
            dataset_name = get_dataset_name(task_config.name, experiment_config.name)
            assert dataset_name not in info
            info[dataset_name] = data

    metashift_factory.save_all(args.output_dir, info)


if __name__ == "__main__":
    main()
