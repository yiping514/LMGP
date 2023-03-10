library(LVGP)
args = commandArgs(trailingOnly=TRUE)
save_dir <- args[1]

ind_qual <- as.integer(read.csv(file.path(save_dir,'../../','qual_index.csv'),header=FALSE)[,1]+1)
test_x <- as.matrix(read.table(file.path(save_dir,'../../','test_x.csv'),header=FALSE,sep=' '))
test_y <- as.matrix(read.csv(file.path(save_dir,'../../','test_y.csv'),header=FALSE))

train_x <- as.matrix(read.table(file.path(save_dir,'train_x.csv'),header=FALSE,sep=' '))
train_y <- as.matrix(read.csv(file.path(save_dir,'train_y.csv'),header=FALSE))

test_x[,ind_qual] <- test_x[,ind_qual]+1
train_x[,ind_qual] <- train_x[,ind_qual]+1

ptm <- proc.time()
model <- LVGP_fit(train_x,train_y,ind_qual = ind_qual)
total_time <- proc.time()-ptm

test_pred <- LVGP_predict(test_x,model)$Y_hat
rrmse <- sqrt(sum((test_y-test_pred)^2)/sum((test_y-mean(test_y))^2))

stats <- data.frame(rrmse = c(rrmse),fit_time=total_time[3])
write.csv(stats,file = file.path(save_dir,'R_stats.csv'),row.names = F)