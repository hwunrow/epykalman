check_param_in_ci <- function(dt, dd, last_epi=FALSE, percentile = 95, colname=NULL) {
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
    if (max(dt[window==max(dt$window), day]) < dd) {
      # day is tooo late
      dd <- max(dt[window==max(dt$window), day])
    }
    percentile_dt <- dt[day==dd, as.list(quantile(.SD, quantiles, na.rm=TRUE)),  .SDcols=paste0("sample", 1:300), by=.(window)]
    truth_dt <- dt[day==dd, .(window, rt)]
    percentile_dt <- merge(percentile_dt, truth_dt, by=c("window"))
    if (last_epi) {
      percentile_dt[, in_ci := `2.5%` <= rt & rt <= `97.5%`]
      names(percentile_dt)[names(percentile_dt) == 'in_ci'] <- colname
      return(percentile_dt[, .SD, .SDcols = c("window", colname)])
    } else {
      percentile_dt[, in_ci := `2.5%` <= rt & rt <= `97.5%`]
      names(percentile_dt)[names(percentile_dt) == 'in_ci'] <- colname
      return(percentile_dt[, .SD, .SDcols = c("window", colname)])
    }
  }
}

compute_ens_var <- function(dt, dd, last_epi=FALSE, colname=NULL) {
  samplecols <- paste0("sample", 1:300)
  if (dd == "last") {
    last_day <- max(dt[window==20, day])
    return(dt[day==last_day, .(ens_var_last_day=rowVars(as.matrix(.SD), na.rm=TRUE)),
              .SDcols=samplecols, by=window])
  } else {
    if (!is.numeric(dd)) {
      stop("day must be numeric")
    }
    if (nrow(dt[day==dd]) < 20) {
      # day is tooo late
      dd <- max(dt[window==20, day])
    }
    return_dt <- dt[day==dd, rowVars(as.matrix(.SD), na.rm=TRUE), .SDcols=samplecols, by=window]
    setnames(return_dt, "V1", colname)
    return(return_dt)
  }
}

bin_data <- function(d, d_pp, num_bins, bins = NULL) {
  data_min <- min(c(min(d), min(d_pp)))
  data_max <- max(c(max(d), max(d_pp)))
  
  if (is.null(bins)) {
    bins <- seq(data_min, data_max, length.out = num_bins)
  }
  
  digitized <- findInterval(d, bins)
  digitized_pp <- findInterval(d_pp, bins)
  
  return(list(digitized, digitized_pp, bins))
}

compute_probs <- function(digitized_data, num_bins) {
  counts <- table(factor(digitized_data, levels = 1:num_bins, exclude = NULL))
  probs <- counts / length(digitized_data)
  return(probs)
}

kl_divergence <- function(p_sample, q_sample, epsilon=1e-10, num_bins=10) {
  stopifnot(length(p_sample) == length(q_sample))
  
  bin_data_result <- bin_data(p_sample, q_sample, num_bins)
  p_bins <- bin_data_result[[1]]
  q_bins <- bin_data_result[[2]]
  bins <- bin_data_result[[3]]
  
  p_probs <- compute_probs(p_bins, num_bins)
  q_probs <- compute_probs(q_bins, num_bins)
  
  p_safe = p_probs + epsilon
  q_safe = q_probs + epsilon
  
  return(sum(p_safe * log(p_safe / q_safe)))
}

avg_kl_divergence <- function(
    dt,
    synthetic_dt,
    data_dt,
    i_ppc,
    dd = NULL,
    evaluate_on = FALSE,
    min_i = 100,
    num_bins = 10,
    colname=NULL) {
  days <- synthetic_dt[i >= min_i, day]
  
  if (!is.null(dd)) {
    dd <- min(dd, max(dt[window==20, day]))
    days <- days[days <= dd]
  }
  
  if (evaluate_on) {
    days = c(dd)
  }
  
  samplecols <- paste0("sample", 1:300)
  
  avg_kl_window <- function(w) {
    tmp_list <- c()
    for (t in days) {
      # Calculate kl
      kl <- kl_divergence(data_dt[day == t, ..samplecols],
                         i_ppc[day == t & window == w, ..samplecols],
                         num_bins = num_bins)
      tmp_list <- c(tmp_list, kl)
    }
    return(data.table(window = w, avg_kl = mean(tmp_list)))
  }
  
  dt_return <- rbindlist(lapply(unique(dt$window), avg_kl_window))
  
  names(dt_return)[names(dt_return) == 'avg_kl'] <- colname
  return(dt_return)
}


wasserstein2 <- function(p_sample, q_sample, num_bins = 10) {
  stopifnot(length(p_sample) == length(q_sample))
  
  bin_data_result <- bin_data(p_sample, q_sample, num_bins)
  p_bins <- bin_data_result[[1]]
  q_bins <- bin_data_result[[2]]
  bins <- bin_data_result[[3]]
  
  p_probs <- compute_probs(p_bins, num_bins)
  q_probs <- compute_probs(q_bins, num_bins)
  
  stopifnot(length(p_probs) == length(q_probs))
  # stopifnot(sum(p_probs) == 1.0, paste("Dist p must sum to 1", sum(p_probs)))
  # stopifnot(sum(q_probs) == 1.0, paste("Dist q must sum to 1", sum(q_probs)))
  
  # Compute the cumulative sums
  p_cdf <- cumsum(p_probs)
  q_cdf <- cumsum(q_probs)
  
  # Calculate the squared distances between the cumulative sums
  squared_distances <- (p_cdf - q_cdf)^2
  
  # Compute the W-2 metric
  w2 <- sqrt(sum(squared_distances))
  
  return(w2)
}


avg_wasserstein2 <- function(dt, synthetic_dt, data_dt, i_ppc, dd=NULL, evaluate_on=FALSE, min_i = 100, num_bins = 10, colname=NULL) {
  days <- synthetic_dt[i >= min_i, day]
  if (!is.null(dd)) {
    dd <- min(dd, max(dt[window==20, day]))
    days <- days[days <= dd]
  }
  
  if (evaluate_on) {
    days = c(dd)
  }
  
  samplecols <- paste0("sample", 1:300)
  
  avg_w2_window <- function(w) {
    tmp_list <- c()
    for (t in days) {
      # Calculate w2
      w2 <- wasserstein2(data_dt[day == t, ..samplecols],
                         i_ppc[day == t & window == w, ..samplecols],
                         num_bins = num_bins)
      tmp_list <- c(tmp_list, w2)
    }
    return(data.table(window = w, avg_w2 = mean(tmp_list)))
  }
  dt_return <- rbindlist(lapply(unique(dt$window), avg_w2_window))
  
  names(dt_return)[names(dt_return) == 'avg_w2'] <- colname
  return(dt_return)
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

data_rmse <- function(dt, synthetic_dt, dd=NULL, last_epi=FALSE, colname=NULL) {
  # generate i from dt
  synthetic_dt$day <- 1:nrow(synthetic_dt)
  samplecols <- paste0("sample", 1:300)
  beta_dt <- dt[, ..samplecols] * gamma
  beta_dt$day <- dt$day
  beta_dt$window <- dt$window
  i_ppc <- free_sim(beta_dt, synthetic_dt)
  i_dt <- merge(i_ppc, synthetic_dt[, .(day,i)], by="day")

  # compute rmse
  i_dt$sqdiff <- rowMeans((i_dt$i - i_dt[, samplecols, with=FALSE])^2)
  
  if (last_epi) {
    result_dt <- i_dt[day<=dd, sqrt(mean(sqdiff, na.rm=TRUE)), by=window]
    setnames(result_dt, "V1", colname)
    return(result_dt)
  }
  
  return(list(rmse_dt = i_dt[, .(data_rmse=sqrt(mean(sqdiff, na.rm=TRUE))), by=window], i_ppc = i_ppc))
}

rt_rmse <- function(dt, dd=NULL, evaluate_on=FALSE, colname=NULL) {
  dt_copy <- copy(dt)
  samplecols <- paste0("sample", 1:300)
  dt_copy$sqdiff <- rowMeans((dt_copy$rt - dt_copy[, samplecols, with=FALSE])^2)
  
  if (is.null(dd)) {
    # average across entire time series
    return(dt_copy[, .(colname=sqrt(mean(sqdiff, na.rm=TRUE))), by=window])
  }
  
  # day dd should be smaller than the last eval in the window
  dd <- min(dd, max(dt_copy[window==20, day]))
  
  if (evaluate_on) {
    # compute rmse on dd
    result_dt <- dt_copy[day == dd, sqrt(mean(sqdiff, na.rm=TRUE)), by=window]
    setnames(result_dt, "V1", colname)
    return(result_dt)
  }
  else {
    # compute rmse from day 1 until dd
    result_dt <- dt_copy[day <= dd, sqrt(mean(sqdiff, na.rm=TRUE)), by=window]
    setnames(result_dt, "V1", colname)
    return(result_dt)
  }
}

compute_crps <- function(dt, dd, colname, ww=20) {
  # Calculate the Continuous Ranked Probability Score (CRPS).
  # Evaluates on day.
  
  # Parameters:
  #   dt (data.table): The EpiEstim posterior samples
  #   dd (int): The day for which to compute the crps
  #   colname (string): column name for output data table
  #   ww (int): The window to check for the day and obs cutoff
  
  # Returns:
  #   crps_dt (data.table): The CRPS score by window
  
  dt_copy <- copy(dt)
  # Handle day out of bounds
  dd <- min(dd, max(dt_copy[window==ww, day]))
  
  # get observation and ensemble
  samplecols <- paste0("sample", 1:300)
  ensembles <- dt_copy[day==dd, samplecols, with=FALSE]
  ensembles <- as.numeric(ensembles)
  observation <- dt_copy[day==dd & window==ww, i]
  
  hist_result <- hist(ensembles, breaks = length(unique(ensembles)), plot = FALSE)
  bins <- hist_result$breaks
  hist <- hist_result$counts
  cdf <- cumsum(hist / length(ensembles))
  
  heaviside <- function(x) {
    if (x > 0) {
      return(1)
    } else if (x == 0) {
      return(0.5)
    } else {
      return(0)
    }
  }
  heaviside <- Vectorize(heaviside)
  
  crps_scores <- (cdf - heaviside(bins - observation)[2:length(bins)])^2 * diff(bins)
  # CRPS scores must be non-negative
  stopifnot(all(crps_scores >= 0.0)) 
  crps_score <- sum(crps_scores)
  
  crps_dt <- data.table(window=unique(dt_copy$window), crps=crps_score)
  setnames(crps_dt, "crps", colname)
  
  return(crps_dt)
}

