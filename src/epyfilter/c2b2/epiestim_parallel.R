################################################################################
#' @Author: Han Yong Wunrow (nhw2114)
#' @Description: Run EpiEstim on C2B2
################################################################################
pacman::p_load(argparse,data.table)

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
pickles <- fread(file.path(in_dir, "pickle_list.csv"))
file <- parameters[task_id, filepath]

# reack in synthetic data pickle file
source_python("open_pickle_rt.py")
pickle_data <- open_pickle_rt(file)
