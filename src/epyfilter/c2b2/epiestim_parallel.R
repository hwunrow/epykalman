################################################################################
#' @Author: Han Yong Wunrow (nhw2114)
#' @Description: Run EpiEstim on C2B2
################################################################################
library(argparse, lib.loc = "filepath")
library(data.table)

# Get arguments from parser
parser <- ArgumentParser()
parser$add_argument("--date", help = "timestamp of current run (i.e. 2014_01_17)", default = NULL, type = "character")
parser$add_argument("--code_dir", help = "code directory", default = NULL, type = "character")
parser$add_argument("--in_dir", help = "directory for external inputs", default = NULL, type = "character")
parser$add_argument("--out_dir", help = "directory for this steps checks", default = NULL, type = "character")

args <- parser$parse_args()
print(args)
list2env(args, environment()); rm(args)

# Get params from parameter map
task_id <- as.integer(Sys.getenv("SGE_TASK_ID"))
parameters <- fread(file.path(code_dir, "check_again.csv"))
location <- parameters[task_id, location_id]