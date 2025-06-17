################################################################################
################################################################################
#' @Author:
#' @DateUpdated: 10/24/2023
#' @Description: This helper function creates and submits a qsub command for the new cluster
#' 
#' @param job_name -N name The name of the job.
#' @param tasks -t number of jobs within array job
#' @param mem  -l mem = Memory usage
#' @param time -l time = Time for your job
#' @param holds -hold_jid list of job names to wait for
#' @param output -o The path used for the standard output stream of the job
#' @param error -e Defines or redefines the path  used  for  the  standard error  stream of the job
#' @param shell R shell file
#' @param script The job's scriptfile
#' @param args Arguments to the job
#'
#' @return NULL
#' 
#' @example: qsub(
#'              job_name = job_name,
#'              mem = mem,
#'              shell_file = shell_file,
#'              script = parallel_script,
#'              args = args
#'            )
################################################################################
################################################################################

qsub <- function(job_name = "array_job",
                 tasks = NULL, 
                 mem = "1G", 
                 time = NULL,
                 holds = NULL, 
                 output = NULL, 
                 error = NULL,
                 shell_file, 
                 script,
                 args = NULL) {
  # check for required arguments
  if (is.null(shell_file)) stop("Did not specify a shell script, please provide one before running again")
  if (is.null(script))     stop("Did not specify a computation script (script parameter), please provide one before running again")
  if (is.null(mem))        stop("Did not specify requested memory, please provide a value for this argument")
  if (is.null(time))       stop("Did not specify requested runtime, please provide a value for this argument")
  
  # format qsub arguments
  my_job                            <- paste("-N", job_name)
  if (!is.null(tasks))   my_tasks   <- paste("-t", tasks, "-tc 500")  else my_tasks   <- "" # -tc limits the number of child jobs that can be running at once
  my_mem                            <- paste0("-l mem=", mem)
  my_time                           <- paste0("time=", time)
  if (!is.null(holds))   my_holds   <- paste("-hold_jid", holds)      else my_holds   <- "" 
  if (!is.null(output))  my_output  <- paste("-o", output)            else my_output  <- ""
  if (!is.null(error))   my_error   <- paste("-e", error)             else my_error   <- ""
  my_shell                          <- paste(shell_file)
  my_code                           <- paste(script)
  if (!is.null(args))    my_args    <- paste(args)                    else my_args    <- ""
  
  my_qsub <-
    paste(
      "qsub",
      my_job,
      my_tasks,
      my_mem,
      my_time,
      my_holds,
      my_output,
      my_error,
      my_shell,
      my_code,
      my_args
    )
  print(my_qsub)
  
  # submit job
  system(my_qsub)
}