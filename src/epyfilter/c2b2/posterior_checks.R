check_param_in_ci <- function(dt, dd, percentile = 95) {
  quantiles <- c((1 - percentile/100)/2, 1 - (1 - percentile/100)/2)
  if (dd == "last") {
    last_day <- max(dt[window==20, day])
    percentile_dt <- dt[day==last_day, as.list(quantile(.SD, quantiles, na.rm=TRUE)),  .SDcols=paste0("sample", 1:300), by=.(window)]
    truth_dt <- dt[day==last_day, .(window, rt)]
    percentile_dt <- merge(percentile_dt, truth_dt, by=c("window"))
    percentile_dt[, in_ci_last_day := `2.5%` <= rt & rt <= `97.5%`]
    return(percentile_dt[, .(window, in_ci_last_day)])
  } else {
    if (!is.numeric(dd)) {
      stop("dd must be numeric")
    }
    percentile_dt <- dt[day==dd, as.list(quantile(.SD, quantiles, na.rm=TRUE)),  .SDcols=paste0("sample", 1:300), by=.(window)]
    truth_dt <- dt[day==dd, .(window, rt)]
    percentile_dt <- merge(percentile_dt, truth_dt, by=c("window"))
    percentile_dt[, in_ci := `2.5%` <= rt & rt <= `97.5%`]
    return(percentile_dt[, .(window, in_ci)])
  }
}

compute_ens_var <- function(dt, dd) {
  samplecols <- paste0("sample", 1:300)
  if (dd == "last") {
    last_day <- max(dt[window==20, day])
    return(dt[day==last_day, .(ens_var_last_day=rowVars(as.matrix(.SD), na.rm=TRUE)),
              .SDcols=samplecols, by=window])
  } else {
    if (!is.numeric(dd)) {
      stop("day must be numeric")
    }
    return(dt[day==dd, .(ens_var=rowVars(as.matrix(.SD), na.rm=TRUE)),
              .SDcols=samplecols, by=window])
  }
}

free_sim <- function(beta_dt, synthetic_dt) {
  m <- 300
  N <- 100000
  S0 <- 99900
  I0 <- 100
  
  samplecols <- paste0("sample", 1:300)
  free_sim_w <- function(w) {
    S <- matrix(rep(S0,m), nrow = 1, ncol = m)
    Ir <- matrix(rep(I0,m), nrow = 1, ncol = m)
    R <- matrix(rep(0, m), nrow = 1, ncol = m)
    i <- matrix(rep(0, m), nrow = 1, ncol = m)
    
    for (t in 1:max(beta_dt[window==w, day])) {
      if (t < 10) {
        beta <- synthetic_dt[day == t, rt] * gamma
        dSI <- rpois(m, beta * Ir[t,] * S[t,] / N)
      } else {
        beta <- as.matrix(beta_dt[window == w & day == t, ..samplecols])
        dSI <- rpois(m, beta * Ir[t,] * S[t,] / N)
      }
      dIR <- rpois(m, Ir[t,] * gamma)
      
      S_new <- pmin(pmax(S[t, ] - dSI, 0), N)
      I_new <- pmin(pmax(Ir[t, ] + dSI - dIR, 0), N)
      R_new <- pmin(pmax(R[t, ] + dIR, 0), N)
      
      S <- rbind(S, S_new)
      Ir <- rbind(Ir, I_new)
      R <- rbind(R, R_new)
      i <- rbind(i, dSI)
    }
    tmp_dt <- data.table(i)
    colnames(tmp_dt) <- samplecols
    tmp_dt$day <- 0:max(beta_dt[window==w, day])
    tmp_dt$window <- w
    
    return(tmp_dt)
  }
  
  dt_list <- lapply(unique(beta_dt$window), free_sim_w)
  i_ppc <- rbindlist(dt_list)
  
  return(i_ppc)
}

data_rmse <- function(dt, synthetic_dt) {
  # generate i from dt
  synthetic_dt$day <- 1:nrow(synthetic_dt)
  samplecols <- paste0("sample", 1:300)
  beta_dt <- dt[, ..samplecols] * gamma
  beta_dt$day <- dt$day
  beta_dt$window <- dt$window
  i_dt <- free_sim(beta_dt, synthetic_dt)
  i_dt <- merge(i_dt, synthetic_dt[, .(day,i)], by="day")

  # compute rmse
  i_dt$sqdiff <- rowMeans((i_dt$i - i_dt[, samplecols, with=FALSE])^2)
  return(i_dt[, .(data_rmse=sqrt(mean(sqdiff, na.rm=TRUE))), by=window])
}


rt_rmse <- function(dt, peaks=NULL) {
  dt_copy <- copy(dt)
  samplecols <- paste0("sample", 1:300)
  dt_copy$sqdiff <- rowMeans((dt_copy$rt - dt_copy[, samplecols, with=FALSE])^2)
  if (is.null(peaks)) {
    return(dt_copy[, .(rt_rmse=sqrt(mean(sqdiff, na.rm=TRUE))), by=window])
  } else {
    if (all(peaks %in% unique(dt_copy$day))) {
      # all peaks in dt
      return(dt_copy[day %in% peaks, .(rt_peak_rmse=sqrt(mean(sqdiff, na.rm=TRUE))), by=window])
    } else if (peaks[1] %in% unique(dt_copy$day)) {
      # first peak in dt
      return(dt_copy[day == peaks[1], .(rt_peak_rmse=sqrt(mean(sqdiff, na.rm=TRUE))), by=window])
    } else {
      return(NA)
    }
  }
}

