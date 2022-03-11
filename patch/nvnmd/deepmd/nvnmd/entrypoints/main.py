
import argparse 

from deepmd.nvnmd.entrypoints.fiodic import fiodic
from deepmd.nvnmd.entrypoints.code import code
from deepmd.nvnmd.entrypoints.debug import debug
from deepmd.nvnmd.entrypoints.vcs import vcs
from deepmd.nvnmd.entrypoints.datacheck import datacheck


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="NVNMD: debug code",
        add_help = False, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # get command
    subparser = parser.add_subparsers(title="Valid subcommands", dest="command")
    # parent parser, whose the options will be added to its children parsers 
    parser_log = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_log.add_argument(
        "-v",
        "--log-level",
        choices=["DEBUG", "3", "INFO", "2", "WARNING", "1", "ERROR", "0"],
        default="INFO",
        help="set verbosity level by string or number, 0=ERROR, 1=WARNING, 2=INFO "
        "and 3=DEBUG",
    )
    parser_log.add_argument(
        "-l",
        "--log-path",
        type=str,
        default=None,
        help="set log file to log messages to disk, if not specified, the logs will "
        "only be output to console",
    )
    # datacheck
    parser_code = subparser.add_parser(
        "datacheck",
        parents=[parser_log],
        help = "check the dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_code.add_argument(
        "-d", "--datapath", 
        type=str,
        default="./",
        help = "the path contains dataset (such as coord.npy, box.npy)",
    )
    parser_code.add_argument(
        "-r", "--rcut", 
        type=str,
        default="6.0",
        help = "the cutoff radius",
    )
    # code
    parser_code = subparser.add_parser(
        "code",
        parents=[parser_log],
        help = "code the verilog code",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_code.add_argument(
        "-d", "--dictionary", 
        type=str,
        default="nvnmd/config.npy",
        help = "the dictionary file contains the key-value pairs",
    )
    parser_code.add_argument(
        "-t", "--template", 
        type=str,
        default="code_tmp/params.v",
        help = "the template file",
    )
    parser_code.add_argument(
        "-o", "--output", 
        type=str,
        default="src/params.v",
        help = "the output file",
    )
    # fiodic
    parser_fiodic = subparser.add_parser(
        "fiodic",
        parents=[parser_log],
        help = "change dic file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_fiodic.add_argument(
        "-i", "--load-file", 
        type=str,
        default="nvnmd/config.npy",
        help = "the loading file",
    )
    parser_fiodic.add_argument(
        "-u", "--update-file", 
        type=str,
        default="nvnmd/config.npy",
        help = "the updating file",
    )
    parser_fiodic.add_argument(
        "-o", "--save-file", 
        type=str,
        default="nvnmd/config.npy",
        help = "the saving file",
    )
    # debug
    parser_debug = subparser.add_parser(
        "debug",
        parents=[parser_log],
        help = "output the debug message for vcs debug",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_debug.add_argument(
        "-c",
        "--nvnmd-config",
        type=str,
        default="nvnmd/config.npy",
        help="the configuration file",
    )
    parser_debug.add_argument(
        "-w",
        "--nvnmd-weight",
        type=str,
        default="nvnmd/weight.npy",
        help="the weight file",
    )
    parser_debug.add_argument(
        "-m",
        "--nvnmd-map",
        type=str,
        default="nvnmd/map.npy",
        help="the file containing the mapping tables",
    )
    parser_debug.add_argument(
        "-a",
        "--atoms-file",
        type=str,
        default="atoms.xsf",
        help="a file containing the atoms structure",
    )
    parser_debug.add_argument(
        "-t",
        "--type-map",
        type=str,
        default="type_map.raw",
        help="a file containing chemical species mapping table",
    )
    parser_debug.add_argument(
        "-d",
        "--nvnmd-debug",
        type=str,
        default="nvnmd/debug/",
        help="a path containing the debug files",
    )
    # vcs
    parser_vcs = subparser.add_parser(
        "vcs",
        parents=[parser_log],
        help = "generate the input file for vcs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_vcs.add_argument(
        "-c",
        "--nvnmd-config",
        type=str,
        default="nvnmd/config.npy",
        help="the configuration file",
    )
    parser_vcs.add_argument(
        "-d",
        "--nvnmd-debug",
        type=str,
        default="nvnmd/debug/res.npy",
        help="the result file of debug command",
    )
    parser_vcs.add_argument(
        "-m",
        "--nvnmd-model",
        type=str,
        default="nvnmd/model.pb",
        help="the binary model file of wrap command",
    )
    parser_vcs.add_argument(
        "-o",
        "--nvnmd-vcs",
        type=str,
        default="nvnmd/vcs",
        help="the output path for vcs command",
    )

    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()
    
    return parsed_args
    
def main():
    args = parse_args()
    dict_args = vars(args)
    if args.command == "datacheck":
        datacheck(**dict_args)
    if args.command == "code":
        code(**dict_args)
    if args.command == "debug":
        debug(**dict_args)
    if args.command == "fiodic":
        fiodic(**dict_args)
    if args.command == "vcs":
        vcs(**dict_args)












