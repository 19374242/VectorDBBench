from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,

)
from vectordb_bench.backend.clients import DB
DBTYPE = DB.VikingDB

class VikingDBTypedDict(TypedDict):
    host: Annotated[
        str, click.option("--host", type=str, help="host", required=True)
    ]
    region: Annotated[
        str, click.option("--region", type=str, help="region", required=True)
    ]
    ak: Annotated[
        str, click.option("--ak", type=str, help="ak", required=True)
    ]
    sk: Annotated[
        str, click.option("--sk", type=str, help="sk", required=True)
    ]
    scheme: Annotated[
        str, click.option("--scheme", type=str, help="scheme", required=True)
    ]

class VikingDBAutoIndexTypedDict(CommonTypedDict, VikingDBTypedDict):
    ...

@cli.command()
@click_parameter_decorators_from_typed_dict(VikingDBAutoIndexTypedDict)
def VikingDBAutoIndex(**parameters: Unpack[VikingDBAutoIndexTypedDict]):
    from .config import VikingDBConfig, AutoIndexConfig

    run(
        db=DBTYPE,
        db_config=VikingDBConfig(
            host=parameters["host"],
            region=parameters["region"],
            ak=parameters["ak"],
            sk=parameters["sk"],
            scheme=parameters["scheme"],
        ),
        db_case_config=AutoIndexConfig(),
        **parameters,
    )