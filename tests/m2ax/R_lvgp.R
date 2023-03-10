library(LVGP)
library(dplyr)

args = commandArgs(trailingOnly=TRUE)
dat <- readxl::read_excel('M2AX_data.xls')
columns <- paste(c('M','A','X'),'site element',sep='-')
dat <- dat %>% mutate_at(.,columns,function(x) as.numeric(as.factor(x)))

save_dir <- args[1]
response <- args[2]

if(response == 'Young'){
  target = "E (Young's modulus)"
} else if (response == 'Shear'){
  target = "G (Shear modulus)"
} else if (response == 'Bulk'){
  target = "B (Bulk modulus)"
}

train_idxs <- as.integer(read.csv(file.path(save_dir,'train_idxs.csv'),header=FALSE)[,1])+1

all_x <- as.matrix(dat[columns])
all_y <- as.matrix(dat[target])

train_x <- all_x[train_idxs,];test_x <- all_x[-train_idxs,]
train_y <- all_y[train_idxs,,drop=F];test_y <- all_y[-train_idxs,,drop=F]

ptm <- proc.time()
model <- LVGP_fit(train_x,train_y,ind_qual = c(1,2))
total_time <- proc.time()-ptm

test_pred <- LVGP_predict(test_x,model)$Y_hat
rrmse <- sqrt(sum((test_y-test_pred)^2)/sum((test_y-mean(test_y))^2))

stats <- data.frame(rrmse = c(rrmse),fit_time=total_time[3])
write.csv(stats,file = file.path(save_dir,'R_stats.csv'),row.names = F)