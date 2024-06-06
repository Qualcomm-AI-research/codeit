# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Any

import hydra

from codeit.task import make_tasks


@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(config: Any) -> None:
    make_tasks(config)


if __name__ == "__main__":
    main()
